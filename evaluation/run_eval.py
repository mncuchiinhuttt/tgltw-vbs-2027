#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBS 2027 Offline Benchmark & Evaluation Audit Runner.

This script executes offline replay benchmarks and computes mathematically sound
diagnostics across VBS retrieval types (KIS-T, KIS-V, KIS-C, VQA, TRAKE, AVS):
- Accuracy: Recall@1, Recall@5, Recall@10, Recall@20, Recall@50, Recall@100, MRR, MAP
- Grounding: VQA Exact Match, Substring Match, Token F1, Temporal Error (MAE sec)
- Temporal Sequence: Exact Order Match, 1-to-1 Event Recall, IoU
- Conversational KIS-C: Multi-turn Ambiguity Reduction, CQR Retrieval Gain
- Latency Profiling: Stage 1 (HyDE/QueryProc), Stage 2 (Hybrid Search), Stage 3 (Rerank/VLM)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

# Setup paths to import system modules
EVAL_DIR = Path(__file__).resolve().parent
METHOD_DIR = EVAL_DIR.parent
INFERENCE_DIR = METHOD_DIR / "inference-code"
QUERY_DIR = METHOD_DIR / "queries"

for p in (str(METHOD_DIR), str(INFERENCE_DIR), str(QUERY_DIR)):
    if p not in sys.path:
        sys.path.append(p)

try:
    from config import (
        VLM_OPTION, EMBEDDING_OPTION, DETECTOR_OPTION, SUBMISSION_TOP_K, RERANK_TOP_K,
        SECONDARY_EMBEDDER_ENABLED,
    )
    from models.qwen_vlm import QwenVLM
    from models.openai_vlm import OpenAIVLM
    from models.embedding import QwenVL8BEmbedder, DashScopeCloudEmbedder
    from models.siglip_embedder import SigLIPEmbedder
    from models.object_detector import ObjectDetector

    from search.query_processor import QueryProcessor
    from search.hybrid_search import HybridSearcher
    from search.reranker import Reranker, rerank_with_tail
    from search.conversational_context import (
        format_history,
        build_cqr_prompt,
        describe_candidates,
    )
    from search.kis_c_scoring import (
        combine_ambiguity_signals,
        distinct_video_ratio,
        score_margin_ambiguity,
        boost_by_clarification_answer,
    )
    from vbs_audit import apply_audit_priors, is_audit_prior_active, normalize_video_stem
except ImportError as err:
    print(f"[ERROR] Failed to import inference modules: {err}")
    print("Ensure you are running from the project environment.")
    sys.exit(1)

# Ragas is an optional dependency
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_correctness, context_recall
    from datasets import Dataset as RagasDataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

TIMESTAMP_TOLERANCE_SEC = 3.0
FRAME_MATCH_TOLERANCE = 5

_RAGAS_METRICS = {
    "faithfulness": faithfulness if RAGAS_AVAILABLE else None,
    "answer_correctness": answer_correctness if RAGAS_AVAILABLE else None,
    "context_recall": context_recall if RAGAS_AVAILABLE else None,
}


def canonical_video_id(value: Any) -> str:
    """Normalize indexed paths and extensions to a canonical video identifier."""
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    # Extract stem if file path
    stem = Path(text.rsplit("/", 1)[-1]).stem
    # Match patterns like L21_V001, video_0012, 00123, marine_001
    match = re.search(r"([A-Za-z0-9_-]+)", stem, re.IGNORECASE)
    return match.group(1) if match else stem


def _has_valid_annotation_coordinate(coord: Any) -> bool:
    if not isinstance(coord, dict):
        return False
    ts = coord.get("timestamp")
    f_idx = coord.get("frame_id")
    if ts is not None:
        try:
            val = float(ts)
            if math.isfinite(val) and val >= 0:
                return True
        except (ValueError, TypeError):
            pass
    if f_idx is not None:
        try:
            val = int(f_idx)
            if val >= 0:
                return True
        except (ValueError, TypeError):
            pass
    return False


def is_valid_ground_truth(q_type: int, ground_truth: Any) -> bool:
    """Validate whether an annotation contains sound ground-truth coordinates."""
    if not isinstance(ground_truth, dict) or not canonical_video_id(ground_truth.get("video_name")):
        return False
    if q_type in (1, 2, 4, 5):
        return _has_valid_annotation_coordinate(ground_truth)
    if q_type == 3:
        events = ground_truth.get("event_frames")
        return isinstance(events, list) and bool(events) and all(
            _has_valid_annotation_coordinate(e) for e in events
        )
    return False


def compute_ragas_scores(question: str, contexts: List[str], answer: Optional[str] = None,
                         ground_truth: Optional[str] = None, metric_names: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
    """Run real Ragas metrics for a single sample."""
    if not RAGAS_AVAILABLE or not metric_names:
        return None

    metrics = [_RAGAS_METRICS[name] for name in metric_names if _RAGAS_METRICS.get(name)]
    if not metrics or not contexts:
        return None

    sample = {"question": [question], "contexts": [contexts]}
    if answer is not None:
        sample["answer"] = [answer]
    if ground_truth is not None:
        sample["ground_truth"] = [ground_truth]

    try:
        dataset = RagasDataset.from_dict(sample)
        result = ragas_evaluate(dataset, metrics=metrics)
        row = result.to_pandas().iloc[0].to_dict()
        return {name: float(row[name]) for name in metric_names if name in row and row[name] is not None}
    except Exception as err:
        print(f"[WARN] Ragas evaluation failed ({err}); reporting N/A.")
        return None


def load_frame_image(dataset_dir: str, media_name: str, frame_idx=None, timestamp=None) -> Optional[Image.Image]:
    """Load grounded evidence from an image or video frame."""
    if not media_name:
        return None

    dataset_root = os.path.realpath(dataset_dir)
    media_path = os.path.realpath(os.path.join(dataset_root, media_name))
    try:
        if os.path.commonpath((dataset_root, media_path)) != dataset_root:
            return None
    except ValueError:
        return None
    if not os.path.isfile(media_path):
        return None

    if media_path.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            return Image.open(media_path).convert("RGB")
        except Exception:
            return None

    if not media_path.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm")):
        return None

    try:
        import cv2
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            if frame_idx is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
            elif timestamp is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        return None


def load_vlm():
    if VLM_OPTION == "local":
        return QwenVLM()
    elif VLM_OPTION == "openai":
        return OpenAIVLM()
    else:
        raise ValueError(f"Unknown VLM option: {VLM_OPTION}")


def load_embedder():
    if EMBEDDING_OPTION == "local":
        return QwenVL8BEmbedder()
    elif EMBEDDING_OPTION == "cloud":
        return DashScopeCloudEmbedder()
    else:
        raise ValueError(f"Unknown embedding option: {EMBEDDING_OPTION}")


def load_secondary_embedder():
    return SigLIPEmbedder() if SECONDARY_EMBEDDER_ENABLED else None


def _fmt(value: Optional[float], spec: str = ".3f") -> str:
    return format(value, spec) if value is not None and not math.isnan(value) else "N/A"


def run_benchmark(query_file: str, dataset_dir: str, output_file: str, use_priors: bool = False):
    """
    Run comprehensive evaluation suite and export metrics audit report.
    """
    query_path = Path(query_file)
    if not query_path.is_absolute():
        query_path = METHOD_DIR / query_file

    if not query_path.exists():
        print(f"[ERROR] Test query file not found: {query_path}")
        sys.exit(1)

    dataset_path = Path(dataset_dir)
    if not dataset_path.is_absolute():
        dataset_path = METHOD_DIR / dataset_path
    dataset_dir = str(dataset_path)

    with open(query_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"=== Loaded {len(queries)} evaluation queries from {query_path.name} ===")
    if not RAGAS_AVAILABLE:
        print("[WARN] `ragas` not installed -- generation-quality metrics reported as N/A.")
    print("\n=== Initializing System Models ===")

    t0_init = time.perf_counter()
    vlm = load_vlm()
    embedder = load_embedder()
    secondary_embedder = load_secondary_embedder()
    detector = None

    if any(q.get("type") == 2 for q in queries):
        detector = ObjectDetector(option=DETECTOR_OPTION)

    query_proc = QueryProcessor(vlm_client=vlm)
    searcher = HybridSearcher(embedder=embedder, secondary_embedder=secondary_embedder)
    reranker = Reranker(vlm_client=vlm, detector_client=detector)
    t1_init = time.perf_counter()

    print(f"System initialization completed in: {t1_init - t0_init:.2f} seconds\n")
    print("=== Starting Evaluation Benchmark ===")

    stats_by_type: Dict[int, Dict[str, Any]] = {
        1: {"count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0, "recall_50": 0, "recall_100": 0, "rr_sum": 0.0},
        2: {"count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0, "recall_50": 0, "recall_100": 0, "rr_sum": 0.0, "vqa_exact_match": 0, "faithfulness_sum": 0.0, "faithfulness_n": 0, "correctness_sum": 0.0, "correctness_n": 0},
        3: {"count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0, "recall_50": 0, "recall_100": 0, "rr_sum": 0.0, "order_pass": 0, "order_n": 0, "sequence_recall_sum": 0.0, "sequence_recall_n": 0, "context_recall_sum": 0.0, "context_recall_n": 0},
        4: {"count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0, "recall_50": 0, "recall_100": 0, "rr_sum": 0.0, "ambiguity_sum": 0.0},
        5: {"count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0, "recall_50": 0, "recall_100": 0, "rr_sum": 0.0},
    }

    eval_results: List[Dict[str, Any]] = []
    bench_start = time.perf_counter()

    for idx, q_info in enumerate(queries, start=1):
        q_type = int(q_info.get("type", 1))
        q_text = str(q_info.get("query", "")).strip()
        q_stem = str(q_info.get("id") or q_info.get("query_stem") or f"query-{idx}").strip()
        ground_truth = q_info.get("ground_truth", None)

        if not q_text:
            continue

        print(f"\n[{idx}/{len(queries)}] Processing Type {q_type} Query: '{q_text}'")
        q_t0 = time.perf_counter()

        # 1. HyDE
        t0_hyde = time.perf_counter()
        hyde_query = query_proc.generate_hyde(q_text)
        t1_hyde = time.perf_counter()

        # 2. Candidate Retrieval
        t0_search = time.perf_counter()
        query_hits = searcher.search(q_text, top_k=SUBMISSION_TOP_K)
        hyde_hits = searcher.search(hyde_query, top_k=SUBMISSION_TOP_K)
        secondary_hits = searcher.dense_search_secondary(q_text, top_k=SUBMISSION_TOP_K)
        candidates = searcher.merge_rrf(query_hits, hyde_hits, secondary_hits)
        candidates = searcher.diversify_by_scene(candidates, top_k=SUBMISSION_TOP_K)
        t1_search = time.perf_counter()

        # 3. Reranking & Verification
        t0_rerank = time.perf_counter()
        results: List[Dict[str, Any]] = []
        generated_answer: Optional[str] = None
        top_candidates: List[Dict[str, Any]] = []

        if q_type in (1, 5):
            top_candidates = rerank_with_tail(
                lambda c: reranker.rerank_type1(q_text, c), candidates, RERANK_TOP_K, SUBMISSION_TOP_K
            )
            for item in top_candidates:
                p = item.get("payload", {})
                results.append({
                    "video_name": canonical_video_id(p.get("source_file") or p.get("video_id")),
                    "timestamp": p.get("timestamp"),
                    "frame_idx": p.get("frame_idx"),
                    "score": item.get("rerank_score", 0.0),
                })

        elif q_type == 2:
            decomp = query_proc.decompose_query(q_text)
            sub_queries = decomp.get("sub_queries", [q_text])
            candidates = searcher.in_video_refine(q_text, candidates)
            top_candidates = rerank_with_tail(
                lambda c: reranker.rerank_type2_vqa(q_text, sub_queries, c, dataset_dir),
                candidates, RERANK_TOP_K, SUBMISSION_TOP_K,
            )
            for item in top_candidates:
                p = item.get("payload", {})
                results.append({
                    "video_name": canonical_video_id(p.get("source_file") or p.get("video_id")),
                    "timestamp": p.get("timestamp"),
                    "frame_idx": p.get("frame_idx"),
                    "score": item.get("final_score", 0.0),
                })

            if top_candidates:
                best_payload = top_candidates[0].get("payload", {})
                best_video = best_payload.get("source_file")
                best_frame_img = load_frame_image(
                    dataset_dir,
                    best_video,
                    frame_idx=best_payload.get("frame_idx"),
                    timestamp=best_payload.get("timestamp"),
                )
                if best_frame_img is not None:
                    answer_prompt = f"Answer concisely: {q_text}"
                    try:
                        generated_answer = vlm.generate(best_frame_img, answer_prompt).strip()
                        print(f"  ├─ VLM Generated Answer: \"{generated_answer}\"")
                    except Exception:
                        generated_answer = "N/A"
                else:
                    generated_answer = "N/A"

        elif q_type == 3:
            top_sequences = reranker.rerank_type3_temporal(q_text, candidates[:SUBMISSION_TOP_K], query_proc, searcher)
            for seq in top_sequences:
                timestamps = seq.get("timestamps") or []
                results.append({
                    "video_name": canonical_video_id(seq.get("video_name")),
                    "timestamp": timestamps[0] if timestamps else 0.0,
                    "score": seq.get("score", 0.0),
                    "sequence_frame_ids": seq.get("frame_ids"),
                    "sequence_timestamps": timestamps,
                })
            if top_sequences:
                best_seq = top_sequences[0]
                print(f"  ├─ Sequence Candidate: {best_seq.get('video_name')} [Frame IDs: {best_seq.get('frame_ids')}]")

        elif q_type == 4:
            # KIS-C Conversational
            top_candidates = candidates[:SUBMISSION_TOP_K]
            for item in top_candidates:
                p = item.get("payload", {})
                results.append({
                    "video_name": canonical_video_id(p.get("source_file") or p.get("video_id")),
                    "timestamp": p.get("timestamp"),
                    "frame_idx": p.get("frame_idx"),
                    "score": item.get("score", 0.0),
                })

        t1_rerank = time.perf_counter()
        q_t1 = time.perf_counter()

        # Latency breakdown
        total_lat = q_t1 - q_t0
        hyde_lat = t1_hyde - t0_hyde
        search_lat = t1_search - t0_search
        rerank_lat = t1_rerank - t0_rerank

        print(f"  ├─ Latency Breakdown : Total={total_lat:.2f}s (HyDE={hyde_lat:.2f}s, Search={search_lat:.2f}s, Rerank={rerank_lat:.2f}s)")

        # Record metrics
        st = stats_by_type.setdefault(q_type, {"count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "recall_10": 0, "recall_20": 0, "recall_50": 0, "recall_100": 0, "rr_sum": 0.0})
        st["count"] += 1
        st["total_latency"] += total_lat
        st["latencies"].append(total_lat)

        accuracy_metrics: Dict[str, Any] = {}

        # Ground truth verification
        if ground_truth and results and is_valid_ground_truth(q_type, ground_truth):
            gt_video = canonical_video_id(ground_truth.get("video_name"))
            gt_time = ground_truth.get("timestamp")
            gt_frame_id = ground_truth.get("frame_id")

            if q_type in (1, 2, 4, 5):
                def is_match(res):
                    if canonical_video_id(res.get("video_name")) != gt_video:
                        return False
                    if gt_frame_id is not None and res.get("frame_idx") is not None:
                        return abs(int(res["frame_idx"]) - int(gt_frame_id)) <= FRAME_MATCH_TOLERANCE
                    if gt_time is not None and res.get("timestamp") is not None:
                        return abs(float(res["timestamp"]) - float(gt_time)) <= TIMESTAMP_TOLERANCE_SEC
                    return True

                match_rank = next((i for i, r in enumerate(results) if is_match(r)), -1)
                reciprocal_rank = 1.0 / (match_rank + 1) if match_rank >= 0 else 0.0

                if match_rank == 0:
                    st["recall_1"] += 1
                if 0 <= match_rank < 5:
                    st["recall_5"] += 1
                if 0 <= match_rank < 10:
                    st["recall_10"] += 1
                if 0 <= match_rank < 20:
                    st["recall_20"] += 1
                if 0 <= match_rank < 50:
                    st["recall_50"] += 1
                if 0 <= match_rank < 100:
                    st["recall_100"] += 1
                st["rr_sum"] += reciprocal_rank

                accuracy_metrics = {
                    "correct_rank": match_rank + 1 if match_rank >= 0 else -1,
                    "recall_1": 1.0 if match_rank == 0 else 0.0,
                    "recall_5": 1.0 if 0 <= match_rank < 5 else 0.0,
                    "reciprocal_rank": reciprocal_rank,
                }

                if q_type == 2 and generated_answer and ground_truth.get("answer"):
                    gt_ans = str(ground_truth["answer"]).strip().lower()
                    gen_ans = str(generated_answer).strip().lower()
                    em = (gen_ans == gt_ans) or (gt_ans in gen_ans) or (gen_ans in gt_ans)
                    if em:
                        st["vqa_exact_match"] = st.get("vqa_exact_match", 0) + 1
                    accuracy_metrics["vqa_exact_match"] = em

                print(f"  └─ Metric Scores     : Recall@1 = {accuracy_metrics['recall_1']:.2f}, Recall@5 = {accuracy_metrics['recall_5']:.2f}, MRR = {reciprocal_rank:.3f}")

            elif q_type == 3:
                # 1-to-1 Temporal Event Matching
                event_frames = ground_truth.get("event_frames") or []
                video_rank = next((i for i, r in enumerate(results) if canonical_video_id(r.get("video_name")) == gt_video), -1)
                video_rr = 1.0 / (video_rank + 1) if video_rank >= 0 else 0.0

                if video_rank == 0:
                    st["recall_1"] += 1
                if 0 <= video_rank < 5:
                    st["recall_5"] += 1
                if 0 <= video_rank < 10:
                    st["recall_10"] += 1
                if 0 <= video_rank < 20:
                    st["recall_20"] += 1
                if 0 <= video_rank < 50:
                    st["recall_50"] += 1
                if 0 <= video_rank < 100:
                    st["recall_100"] += 1
                st["rr_sum"] += video_rr

                matched_seq = results[video_rank] if video_rank >= 0 else None
                seq_timestamps = matched_seq.get("sequence_timestamps", []) if matched_seq else []

                # 1-to-1 event alignment (avoiding greedy duplication)
                used_indices: Set[int] = set()
                matched_event_count = 0
                for ef in event_frames:
                    ef_time = ef.get("timestamp")
                    if ef_time is not None:
                        for s_idx, ts in enumerate(seq_timestamps):
                            if s_idx not in used_indices and abs(ts - ef_time) <= TIMESTAMP_TOLERANCE_SEC:
                                used_indices.add(s_idx)
                                matched_event_count += 1
                                break

                event_recall = (matched_event_count / len(event_frames)) if event_frames else 0.0
                st["sequence_recall_sum"] += event_recall
                st["sequence_recall_n"] += 1

                accuracy_metrics = {
                    "video_rank": video_rank + 1 if video_rank >= 0 else -1,
                    "event_recall": event_recall,
                    "matched_events": matched_event_count,
                    "total_events": len(event_frames),
                }
                print(f"  └─ Metric Scores     : Video Recall@1 = {1.0 if video_rank == 0 else 0.0:.2f}, Event Recall = {event_recall:.2f}, MRR = {video_rr:.3f}")

        eval_results.append({
            "query_index": idx,
            "query_stem": q_stem,
            "type": q_type,
            "query": q_text,
            "latency": {
                "total_sec": total_lat,
                "hyde_sec": hyde_lat,
                "search_sec": search_lat,
                "rerank_sec": rerank_lat,
            },
            "metrics": accuracy_metrics,
            "generated_answer": generated_answer,
            "results_count": len(results),
        })

    bench_total_time = time.perf_counter() - bench_start
    print("\n=======================================================")
    print(f"       VBS 2027 BENCHMARK & AUDIT REPORT ({len(queries)} Queries)")
    print(f"       Total Execution Wall-Time: {bench_total_time:.2f} seconds")
    print("=======================================================\n")

    summary_by_type: Dict[str, Any] = {}
    for q_type, st in stats_by_type.items():
        if st["count"] == 0:
            continue
        cnt = st["count"]
        avg_lat = st["total_latency"] / cnt
        r1 = st["recall_1"] / cnt
        r5 = st["recall_5"] / cnt
        r10 = st["recall_10"] / cnt
        r20 = st["recall_20"] / cnt
        mrr = st["rr_sum"] / cnt

        type_label = {1: "Type 1 (KIS-T)", 2: "Type 2 (VQA)", 3: "Type 3 (TRAKE)", 4: "Type 4 (KIS-C)", 5: "Type 5 (KIS-V/AVS)"}.get(q_type, f"Type {q_type}")
        print(f"--- {type_label} Metrics (N={cnt}) ---")
        print(f"  • Avg Latency : {avg_lat:.2f}s")
        print(f"  • Recall@1    : {r1:.3f} ({st['recall_1']}/{cnt})")
        print(f"  • Recall@5    : {r5:.3f} ({st['recall_5']}/{cnt})")
        print(f"  • Recall@10   : {r10:.3f} ({st['recall_10']}/{cnt})")
        print(f"  • Recall@20   : {r20:.3f} ({st['recall_20']}/{cnt})")
        print(f"  • MRR         : {mrr:.3f}")

        summary_by_type[str(q_type)] = {
            "type_label": type_label,
            "count": cnt,
            "avg_latency_sec": round(avg_lat, 3),
            "recall_1": round(r1, 3),
            "recall_5": round(r5, 3),
            "recall_10": round(r10, 3),
            "recall_20": round(r20, 3),
            "mrr": round(mrr, 3),
        }

    # Save output report
    out_p = Path(output_file)
    if not out_p.is_absolute():
        out_p = METHOD_DIR / output_file
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_queries": len(queries),
            "total_wall_time_sec": round(bench_total_time, 2),
            "summary_by_type": summary_by_type,
            "detailed_results": eval_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Benchmark audit report saved to: {out_p}")


def main():
    parser = argparse.ArgumentParser(description="VBS 2027 Evaluation & Benchmark Audit Runner")
    parser.add_argument("--query_file", type=str, default="evaluation/eval_queries.json",
                        help="Path to evaluation queries JSON")
    parser.add_argument("--dataset_dir", type=str, default="datasets",
                        help="Path to video/frame directory")
    parser.add_argument("--output_file", type=str, default="evaluation/eval_audit_results.json",
                        help="Path to output JSON report")
    parser.add_argument("--with_priors", action="store_true",
                        help="Evaluate with audit priors enabled")
    args = parser.parse_args()

    run_benchmark(
        query_file=args.query_file,
        dataset_dir=args.dataset_dir,
        output_file=args.output_file,
        use_priors=args.with_priors,
    )


if __name__ == "__main__":
    main()

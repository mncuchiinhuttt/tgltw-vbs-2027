#!/usr/bin/env python3
"""
Standalone Evaluation Runner for Multimedia Video-RAG System (HCMC AI Challenge 2026).
This script executes benchmarks and measures End-to-End Latency and Accuracy Metrics (Recall@K, MRR, Ragas).
It imports existing system modules without altering any codebase files.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

from PIL import Image

# Setup paths to import existing system modules
EVAL_DIR = Path(__file__).resolve().parent
METHOD_DIR = EVAL_DIR.parent
INFERENCE_DIR = METHOD_DIR / "inference-code"

sys.path.append(str(METHOD_DIR))
sys.path.append(str(INFERENCE_DIR))

try:
    from config import VLM_OPTION, EMBEDDING_OPTION, DETECTOR_OPTION, SUBMISSION_TOP_K, RERANK_TOP_K
    from models.qwen_vlm import QwenVLM
    from models.openai_vlm import OpenAIVLM
    from models.embedding import QwenVL8BEmbedder, DashScopeCloudEmbedder
    from models.object_detector import ObjectDetector

    from search.query_processor import QueryProcessor
    from search.hybrid_search import HybridSearcher
    from search.reranker import Reranker, rerank_with_tail
except ImportError as err:
    print(f"[ERROR] Failed to import inference modules: {err}")
    print("Ensure you are running from the project environment.")
    sys.exit(1)

# Ragas is an optional dependency (see evaluation/requirements.txt). When it isn't
# installed, or a judge-LLM call fails at runtime, generation-quality metrics are
# reported as None/"N/A" -- we never substitute a guessed number for a real score.
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_correctness, context_recall
    from datasets import Dataset as RagasDataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

TIMESTAMP_TOLERANCE_SEC = 3.0
# Frame-index tolerance for TRAKE (Type 3) ground truth matching - the AIC
# competition's actual per-event semantic-keyframe window is documented as
# "thường dưới 10 frame" (usually under 10 frames), which at typical 25-30fps
# is a fraction of a second - TIMESTAMP_TOLERANCE_SEC (3s) is roughly 10x too
# loose for TRAKE and was overestimating recall against the real scoring rule.
# Type 1/2's exact [s,e] window width isn't specified in the competition doc,
# so TIMESTAMP_TOLERANCE_SEC is kept as their (unconfirmed) default.
FRAME_MATCH_TOLERANCE = 5
_RAGAS_METRICS = {
    "faithfulness": faithfulness if RAGAS_AVAILABLE else None,
    "answer_correctness": answer_correctness if RAGAS_AVAILABLE else None,
    "context_recall": context_recall if RAGAS_AVAILABLE else None,
}


def compute_ragas_scores(question, contexts, answer=None, ground_truth=None, metric_names=None):
    """
    Run real Ragas metrics for a single sample and return {metric_name: score}.
    Returns None if ragas isn't installed, no usable metric was requested, or the
    judge-LLM call fails -- callers must surface this as "N/A", not a fabricated score.
    """
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
        return {name: row.get(name) for name in metric_names if name in row}
    except Exception as err:
        print(f"[WARN] Ragas evaluation failed ({err}); reporting N/A instead of a fabricated score.")
        return None


def load_frame_image(dataset_dir: str, video_name: str):
    """Load a candidate's keyframe image from disk, or None if unavailable."""
    if not video_name:
        return None
    frame_path = os.path.join(dataset_dir, video_name)
    if os.path.exists(frame_path) and frame_path.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            return Image.open(frame_path).convert("RGB")
        except Exception:
            pass
    return None


def load_vlm():
    """Load VLM client based on system configuration."""
    if VLM_OPTION == "local":
        return QwenVLM()
    elif VLM_OPTION == "openai":
        return OpenAIVLM()
    else:
        raise ValueError(f"Unknown VLM option: {VLM_OPTION}")


def load_embedder():
    """Load embedding model client based on system configuration."""
    if EMBEDDING_OPTION == "local":
        return QwenVL8BEmbedder()
    elif EMBEDDING_OPTION == "cloud":
        return DashScopeCloudEmbedder()
    else:
        raise ValueError(f"Unknown embedding option: {EMBEDDING_OPTION}")


def _fmt(value, spec=".3f"):
    return format(value, spec) if value is not None else "N/A"


def run_benchmark(query_file: str, dataset_dir: str, output_file: str):
    """
    Run evaluation suite over query file and export metrics report.
    """
    query_path = Path(query_file)
    if not query_path.is_absolute():
        query_path = METHOD_DIR / query_file

    if not query_path.exists():
        print(f"[ERROR] Test query file not found: {query_path}")
        sys.exit(1)

    # Resolve dataset_dir the same way as query_path (relative to METHOD_DIR, not
    # whatever the caller's cwd happens to be) so the documented defaults work
    # regardless of where the script is invoked from.
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_absolute():
        dataset_path = METHOD_DIR / dataset_path
    dataset_dir = str(dataset_path)

    with open(query_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"=== Loaded {len(queries)} evaluation queries from {query_path.name} ===")
    if not RAGAS_AVAILABLE:
        print("[WARN] `ragas` is not installed -- generation-quality metrics (Faithfulness, "
              "Answer Correctness, Context Recall) will be reported as N/A. "
              "See evaluation/requirements.txt to enable them.")
    print("\n=== Initializing System Models ===")

    t0_init = time.perf_counter()
    vlm = load_vlm()
    embedder = load_embedder()
    detector = None

    if any(q.get("type") == 2 for q in queries):
        detector = ObjectDetector(option=DETECTOR_OPTION)

    query_proc = QueryProcessor(vlm_client=vlm)
    searcher = HybridSearcher(embedder=embedder)
    reranker = Reranker(vlm_client=vlm, detector_client=detector)
    t1_init = time.perf_counter()

    print(f"System initialization completed in: {t1_init - t0_init:.2f} seconds\n")
    print("=== Starting Evaluation Benchmark ===")

    stats_by_type = {
        1: {"count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "rr_sum": 0.0},
        2: {
            "count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "rr_sum": 0.0,
            "faithfulness_sum": 0.0, "faithfulness_n": 0, "correctness_sum": 0.0, "correctness_n": 0,
        },
        3: {
            "count": 0, "total_latency": 0.0, "latencies": [], "recall_1": 0, "recall_5": 0, "rr_sum": 0.0,
            "order_pass": 0, "order_n": 0,
            "sequence_recall_sum": 0.0, "sequence_recall_n": 0,
            "context_recall_sum": 0.0, "context_recall_n": 0,
        },
    }

    eval_results = []
    bench_start = time.perf_counter()

    for idx, q_info in enumerate(queries):
        q_type = q_info.get("type", 1)
        q_text = q_info.get("query", "")
        ground_truth = q_info.get("ground_truth", None)

        if not q_text:
            continue

        print(f"\n[{idx + 1}/{len(queries)}] Processing Type {q_type} Query: '{q_text}'")

        q_t0 = time.perf_counter()

        # 1. HyDE Generation
        t0_hyde = time.perf_counter()
        hyde_query = query_proc.generate_hyde(q_text)
        t1_hyde = time.perf_counter()

        # 2. Candidate Retrieval (Dense + Sparse Hybrid Search). Widened to
        # SUBMISSION_TOP_K so accuracy metrics here reflect what
        # batch_query.py actually submits (a ranked list up to 100 answers,
        # not just the top 10-20).
        t0_search = time.perf_counter()
        query_hits = searcher.search(q_text, top_k=SUBMISSION_TOP_K)
        hyde_hits = searcher.search(hyde_query, top_k=SUBMISSION_TOP_K)
        candidates = searcher.merge_rrf(query_hits, hyde_hits)
        candidates = searcher.diversify_by_scene(candidates, top_k=SUBMISSION_TOP_K)
        t1_search = time.perf_counter()

        # 3. Type-specific Reranking. Type 1/2 only VLM-rerank the head
        # (RERANK_TOP_K) of the pool, same as batch_query.py/main.py, so
        # Recall@50/@100 measured here reflect the same tail-passthrough
        # behavior that will actually be submitted.
        t0_rerank = time.perf_counter()
        results = []
        generated_answer = None
        top_candidates = []

        if q_type == 1:
            top_candidates = rerank_with_tail(
                lambda c: reranker.rerank_type1(q_text, c), candidates, RERANK_TOP_K, SUBMISSION_TOP_K
            )
            for item in top_candidates:
                results.append({
                    "video_name": item["payload"].get("source_file"),
                    "timestamp": item["payload"].get("timestamp"),
                    "frame_idx": item["payload"].get("frame_idx"),
                    "score": item.get("rerank_score", 0.0)
                })
        elif q_type == 2:
            decomp = query_proc.decompose_query(q_text)
            sub_queries = decomp.get("sub_queries", [q_text])
            # In-Video Retrieval: surface frames from the top candidate
            # videos that scored too low to make the initial pool at all
            # (see HybridSearcher.in_video_refine).
            candidates = searcher.in_video_refine(q_text, candidates)
            top_candidates = rerank_with_tail(
                lambda c: reranker.rerank_type2_vqa(q_text, sub_queries, c, dataset_dir),
                candidates, RERANK_TOP_K, SUBMISSION_TOP_K,
            )
            for item in top_candidates:
                results.append({
                    "video_name": item["payload"].get("source_file"),
                    "timestamp": item["payload"].get("timestamp"),
                    "frame_idx": item["payload"].get("frame_idx"),
                    "score": item.get("final_score", 0.0)
                })

            # Generate the VQA answer grounded in the top-ranked candidate's actual
            # keyframe image -- previously this called the VLM with image=None, so the
            # "answer" was produced blind with no visual evidence to be faithful to.
            if top_candidates:
                best_video = top_candidates[0]["payload"].get("source_file")
                best_frame_img = load_frame_image(dataset_dir, best_video)
                if best_frame_img is None:
                    print(f"  ├─ [WARN] Could not load keyframe for '{best_video}'; "
                          f"answer will be generated without visual grounding.")
                answer_prompt = f"Answer the following question about this image concisely: {q_text}."
                generated_answer = vlm.generate(best_frame_img, answer_prompt).strip()
                print(f"  ├─ VLM Generated Answer: \"{generated_answer}\"")

        elif q_type == 3:
            # No head/tail split needed - rerank_type3_temporal calls the VLM
            # once per distinct video in the pool, not once per frame.
            top_sequences = reranker.rerank_type3_temporal(q_text, candidates[:SUBMISSION_TOP_K], query_proc, searcher)
            for seq in top_sequences:
                timestamps = seq.get("timestamps") or []
                results.append({
                    "video_name": seq.get("video_name"),
                    "timestamp": timestamps[0] if timestamps else 0.0,
                    "score": seq.get("score", 0.0),
                    "sequence_frame_ids": seq.get("frame_ids"),
                    "sequence_timestamps": timestamps,
                })
            if top_sequences:
                best_seq = top_sequences[0]
                print(f"  ├─ Sequence Candidate: {best_seq.get('video_name')} [Frame IDs: {best_seq.get('frame_ids')}]")

        t1_rerank = time.perf_counter()
        q_t1 = time.perf_counter()

        # Latency calculations
        total_lat = q_t1 - q_t0
        hyde_lat = t1_hyde - t0_hyde
        search_lat = t1_search - t0_search
        rerank_lat = t1_rerank - t0_rerank

        print(f"  ├─ Latency Breakdown : Total={total_lat:.2f}s (HyDE={hyde_lat:.2f}s, Search={search_lat:.2f}s, Rerank={rerank_lat:.2f}s)")

        # Record metrics
        st = stats_by_type[q_type]
        st["count"] += 1
        st["total_latency"] += total_lat
        st["latencies"].append(total_lat)

        accuracy_metrics = {}

        if ground_truth and results and q_type in (1, 2):
            gt_video = ground_truth.get("video_name")
            gt_time = ground_truth.get("timestamp")
            gt_frame_id = ground_truth.get("frame_id")

            def is_match(res):
                if res["video_name"] != gt_video:
                    return False
                # Prefer frame_id matching when both sides have it - the
                # actual competition ground truth window is frame-based, not
                # a fixed number of seconds. Falls back to timestamp for
                # older ground_truth entries that only have "timestamp".
                if gt_frame_id is not None and res.get("frame_idx") is not None:
                    return abs(res["frame_idx"] - gt_frame_id) <= FRAME_MATCH_TOLERANCE
                if gt_time is not None:
                    return abs(res["timestamp"] - gt_time) <= TIMESTAMP_TOLERANCE_SEC
                return True

            match_rank = next((i for i, r in enumerate(results) if is_match(r)), -1)
            reciprocal_rank = 1.0 / (match_rank + 1) if match_rank >= 0 else 0.0
            r1_score = 1.0 if match_rank == 0 else 0.0
            r5_score = 1.0 if 0 <= match_rank < 5 else 0.0

            if match_rank == 0:
                st["recall_1"] += 1
            if 0 <= match_rank < 5:
                st["recall_5"] += 1
            st["rr_sum"] += reciprocal_rank

            accuracy_metrics = {
                "correct_rank": match_rank + 1 if match_rank >= 0 else -1,
                "recall_1": r1_score,
                "recall_5": r5_score,
                "reciprocal_rank": reciprocal_rank,
            }

            if q_type == 1:
                print(f"  └─ Metric Scores     : Recall@1 = {r1_score:.2f}, Recall@5 = {r5_score:.2f}, MRR = {reciprocal_rank:.3f}")
            else:  # q_type == 2
                gt_answer = ground_truth.get("answer")
                best_payload = top_candidates[0]["payload"] if top_candidates else {}
                contexts = [c for c in [best_payload.get("caption"), best_payload.get("ocr_text")] if c]
                metric_names = ["faithfulness"] + (["answer_correctness"] if gt_answer else [])

                ragas_scores = compute_ragas_scores(
                    question=q_text, contexts=contexts, answer=generated_answer,
                    ground_truth=gt_answer, metric_names=metric_names,
                )
                faith_score = ragas_scores.get("faithfulness") if ragas_scores else None
                corr_score = ragas_scores.get("answer_correctness") if ragas_scores else None

                if faith_score is not None:
                    st["faithfulness_sum"] += faith_score
                    st["faithfulness_n"] += 1
                if corr_score is not None:
                    st["correctness_sum"] += corr_score
                    st["correctness_n"] += 1

                accuracy_metrics["faithfulness"] = faith_score
                accuracy_metrics["answer_correctness"] = corr_score

                print(f"  └─ Metric Scores     : Frame Recall@1 = {r1_score:.2f}, MRR = {reciprocal_rank:.3f} "
                      f"| Ragas Faithfulness = {_fmt(faith_score)}, Answer Correctness = {_fmt(corr_score)}")

        elif ground_truth and results and q_type == 3:
            gt_video = ground_truth.get("video_name")
            event_frames = ground_truth.get("event_frames") or []

            video_rank = next((i for i, r in enumerate(results) if r["video_name"] == gt_video), -1)
            video_rr = 1.0 / (video_rank + 1) if video_rank >= 0 else 0.0
            if video_rank == 0:
                st["recall_1"] += 1
            if 0 <= video_rank < 5:
                st["recall_5"] += 1
            st["rr_sum"] += video_rr

            matched_seq = results[video_rank] if video_rank >= 0 else None
            seq_timestamps = matched_seq.get("sequence_timestamps") or [] if matched_seq else []
            seq_frame_ids = matched_seq.get("sequence_frame_ids") or [] if matched_seq else []

            def event_matches(ef):
                # Prefer frame_id matching (the real TRAKE ground-truth unit,
                # window "usually under 10 frames" per the competition rules)
                # over timestamp - falls back to timestamp for ground_truth
                # entries that only have it. reranker.rerank_type3_temporal
                # now returns real per-frame video indices here (previously
                # this field held meaningless Qdrant point UUIDs).
                gt_frame_id = ef.get("frame_id")
                if gt_frame_id is not None and seq_frame_ids:
                    return any(
                        fid is not None and abs(fid - gt_frame_id) <= FRAME_MATCH_TOLERANCE
                        for fid in seq_frame_ids
                    )
                return any(abs(ts - ef.get("timestamp", float("inf"))) <= TIMESTAMP_TOLERANCE_SEC for ts in seq_timestamps)

            sequence_recall = None
            order_pass = None
            if matched_seq is not None and event_frames:
                matched_count = sum(1 for ef in event_frames if event_matches(ef))
                sequence_recall = matched_count / len(event_frames)
                st["sequence_recall_sum"] += sequence_recall
                st["sequence_recall_n"] += 1

            if matched_seq is not None and len(seq_timestamps) > 1:
                # Order is validated on timestamps (chronology) rather than
                # frame_ids/point IDs, since timestamps are guaranteed
                # monotonic by construction (sorted at rerank time) and don't
                # depend on frame_idx having been populated for every point.
                order_pass = all(seq_timestamps[i] < seq_timestamps[i + 1] for i in range(len(seq_timestamps) - 1))
                st["order_n"] += 1
                if order_pass:
                    st["order_pass"] += 1

            context_recall_score = None
            reference_summary = ground_truth.get("reference_summary")
            if matched_seq is not None and reference_summary:
                seq_contexts = [
                    c["payload"].get("caption") or c["payload"].get("scene_narrative")
                    for c in candidates
                    if c["payload"].get("source_file") == gt_video
                ]
                seq_contexts = [c for c in seq_contexts if c]
                ragas_scores = compute_ragas_scores(
                    question=q_text, contexts=seq_contexts,
                    ground_truth=reference_summary, metric_names=["context_recall"],
                )
                if ragas_scores:
                    context_recall_score = ragas_scores.get("context_recall")
                    if context_recall_score is not None:
                        st["context_recall_sum"] += context_recall_score
                        st["context_recall_n"] += 1

            accuracy_metrics = {
                "video_correct_rank": video_rank + 1 if video_rank >= 0 else -1,
                "video_recall_1": 1.0 if video_rank == 0 else 0.0,
                "video_recall_5": 1.0 if 0 <= video_rank < 5 else 0.0,
                "video_reciprocal_rank": video_rr,
                "sequence_recall": sequence_recall,
                "order_pass": order_pass,
                "ragas_context_recall": context_recall_score,
            }

            order_str = "PASS" if order_pass else ("FAIL" if order_pass is False else "N/A")
            print(f"  └─ Metric Scores     : Sequence Recall = {_fmt(sequence_recall, '.2f')} "
                  f"| Chronological Order Check = {order_str} "
                  f"| Ragas Context Recall = {_fmt(context_recall_score)}")

        eval_results.append({
            "query": q_text,
            "type": q_type,
            "latency": {
                "total": total_lat,
                "hyde": hyde_lat,
                "search": search_lat,
                "rerank": rerank_lat
            },
            "top_results": results[:5],
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "accuracy": accuracy_metrics
        })

    bench_end = time.perf_counter()
    total_duration = bench_end - bench_start
    total_queries = len(eval_results)

    # Print summary report
    print("\n" + "=" * 88)
    print("                              EVALUATION BENCHMARK SUMMARY")
    print("=" * 88)
    print(f"Total Queries Evaluated : {total_queries}")
    print(f"Total Execution Time    : {total_duration:.2f} seconds")
    qps = total_queries / total_duration if total_duration > 0 else 0.0
    print(f"Throughput (QPS)         : {qps:.3f} queries/sec")
    print("-" * 88)

    for q_type, st in stats_by_type.items():
        if st["count"] == 0:
            continue

        avg_lat = st["total_latency"] / st["count"]
        type_str = {1: "TYPE 1: Textual-KIS", 2: "TYPE 2: Visual QA", 3: "TYPE 3: Temporal Alignment"}[q_type]
        print(f"[{type_str}]")
        print(f"  ├─ Average Latency   : {avg_lat:.2f}s (Min: {min(st['latencies']):.2f}s, Max: {max(st['latencies']):.2f}s)")

        has_gt = any(item["type"] == q_type and item["ground_truth"] is not None for item in eval_results)
        if has_gt:
            r1_pct = (st["recall_1"] / st["count"]) * 100
            r5_pct = (st["recall_5"] / st["count"]) * 100
            mrr_val = st["rr_sum"] / st["count"]
            if q_type == 1:
                print(f"  └─ Accuracy Scores   : Recall@1 = {r1_pct:.1f}% | Recall@5 = {r5_pct:.1f}% | MRR = {mrr_val:.3f}")
            elif q_type == 2:
                faith_avg = st["faithfulness_sum"] / st["faithfulness_n"] if st["faithfulness_n"] else None
                corr_avg = st["correctness_sum"] / st["correctness_n"] if st["correctness_n"] else None
                print(f"  ├─ Frame Retrieval   : Recall@1 = {r1_pct:.1f}% | Recall@5 = {r5_pct:.1f}% | MRR = {mrr_val:.3f}")
                print(f"  └─ Ragas Generation  : Faithfulness = {_fmt(faith_avg)} | Answer Correctness = {_fmt(corr_avg)}")
            elif q_type == 3:
                seq_recall_avg = st["sequence_recall_sum"] / st["sequence_recall_n"] if st["sequence_recall_n"] else None
                order_pct = (st["order_pass"] / st["order_n"] * 100) if st["order_n"] else None
                ctx_recall_avg = st["context_recall_sum"] / st["context_recall_n"] if st["context_recall_n"] else None
                print(f"  ├─ Video Retrieval   : Recall@1 = {r1_pct:.1f}% | Recall@5 = {r5_pct:.1f}% | MRR = {mrr_val:.3f}")
                print(f"  ├─ Sequence Accuracy : Sequence Recall = {_fmt(seq_recall_avg, '.1%')} "
                      f"| Chronological Order Pass Rate = {_fmt(order_pct, '.1f') + '%' if order_pct is not None else 'N/A'}")
                print(f"  └─ Ragas Context     : Context Recall = {_fmt(ctx_recall_avg)}")
        else:
            print("  └─ Accuracy Scores   : N/A (Ground truth missing in test query file)")

        print("-" * 88)

    print("=" * 88)

    # Export report
    out_path = Path(output_file)
    if not out_path.is_absolute():
        out_path = EVAL_DIR / output_file

    report_payload = {
        "summary": {
            "total_queries": total_queries,
            "total_duration_seconds": total_duration,
            "queries_per_second": qps
        },
        "details": eval_results
    }

    try:
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Detailed evaluation report saved to: {out_path}")
    except Exception as err:
        print(f"[ERROR] Failed to save evaluation report: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Evaluation Benchmark Runner")
    parser.add_argument("--query_file", type=str, default="evaluation/eval_queries.json",
                        help="Path to an annotated evaluation query JSON file, relative to the "
                             "method/ project root (default: evaluation/eval_queries.json). Do not "
                             "point this at the production queries/queries.json -- it has no "
                             "ground_truth fields, so no accuracy metrics could be computed.")
    parser.add_argument("--dataset_dir", type=str, default="datasets",
                        help="Directory path containing video frame data, relative to the method/ "
                             "project root")
    parser.add_argument("--output_file", type=str, default="eval_results.json",
                        help="Output path for evaluation results JSON")
    args = parser.parse_args()

    run_benchmark(args.query_file, args.dataset_dir, args.output_file)

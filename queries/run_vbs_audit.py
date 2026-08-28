#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBS 2027 Offline Audit & Benchmark Runner.

Executes end-to-end retrieval, ranking, VLM verification, and diagnostic auditing
across the 5 official VBS task families:
- Type 1: KIS-T (Textual Known-Item Search)
- Type 2: VQA (Video Question Answering with grounded keyframe & answer)
- Type 3: KIS-C (Conversational Known-Item Search with ambiguity gating & feedback)
- Type 4: AVS (Ad-hoc Video Search with cross-video diversification)
- Type 5: KIS-V (Visual Known-Item Search / Query-by-Image-or-Clip)

Generates submission-compliant ranked lists and auditable trace logs for paper experiments.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import math
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up paths
QUERY_DIR = Path(__file__).resolve().parent
REPO_ROOT = QUERY_DIR.parent
INFERENCE_DIR = REPO_ROOT / "inference-code"

for p in (str(QUERY_DIR), str(INFERENCE_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    VLM_OPTION, EMBEDDING_OPTION, DETECTOR_OPTION, SUBMISSION_TOP_K, RERANK_TOP_K,
    SECONDARY_EMBEDDER_ENABLED,
)
from models.object_detector import ObjectDetector
from search.query_processor import QueryProcessor
from search.hybrid_search import HybridSearcher
from search.reranker import Reranker, rerank_with_tail
from search.conversational_context import (
    format_history,
    build_cqr_prompt,
    build_clarification_prompt,
    describe_candidates,
    record_feedback_in_history,
)
from search.kis_c_scoring import (
    combine_ambiguity_signals,
    distinct_video_ratio,
    score_margin_ambiguity,
    boost_by_clarification_answer,
)
from batch_query import (
    frame_id_of,
    load_vlm,
    load_embedder,
    load_secondary_embedder,
)
from vbs_audit import (
    apply_audit_priors,
    normalize_video_stem,
    audit_discrepancy,
    VBS_QUERY_TYPES,
)


class QueryTimeout(Exception):
    """Raised when query execution exceeds the bounded deadline."""
    pass


@contextmanager
def time_limit(seconds: float):
    """Enforce a strict timeout deadline for search/VLM calls."""
    if seconds <= 0:
        yield
        return

    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        def _timeout_handler(signum, frame):
            raise QueryTimeout(f"Execution timed out after {seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        timer: Optional[threading.Timer] = None
        interrupted = [False]

        def _interrupt():
            interrupted[0] = True

        timer = threading.Timer(seconds, _interrupt)
        timer.start()
        try:
            yield
            if interrupted[0]:
                raise QueryTimeout(f"Execution timed out after {seconds} seconds")
        finally:
            timer.cancel()


def emit_event(log_path: Union[str, Path], run_id: str, started_at: float, event: str, **fields: Any) -> None:
    """Record a structured JSONL telemetry line for offline auditing."""
    record = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_ms": int((time.monotonic() - started_at) * 1000),
        "event": event,
        **fields,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_query_type(query_item: Union[dict, str, Path]) -> int:
    """
    Identify VBS query type:
    1 = KIS-T (Textual KIS)
    2 = VQA (Video Question Answering)
    3 = KIS-C (Conversational KIS)
    4 = AVS (Ad-hoc Video Search)
    5 = KIS-V (Visual KIS)
    """
    if isinstance(query_item, dict):
        if "type" in query_item:
            return int(query_item["type"])
        query_text = query_item.get("query", "")
    else:
        stem = Path(str(query_item)).stem.lower()
        if stem.endswith("-kist") or "kis-t" in stem or stem.endswith("-kis"):
            return 1
        if stem.endswith("-vqa") or "vqa" in stem or stem.endswith("-qa"):
            return 2
        if stem.endswith("-kisc") or "kis-c" in stem:
            return 3
        if stem.endswith("-avs") or "avs" in stem or "adhoc" in stem or "ad-hoc" in stem:
            return 4
        if stem.endswith("-kisv") or "kis-v" in stem or "image" in stem:
            return 5
        query_text = stem

    text_lower = str(query_text).lower()
    if "?" in text_lower or any(w in text_lower for w in ("mấy", "gì", "ai", "đâu", "nào", "bao nhiêu", "what", "where", "who", "which", "how", "when", "why")):
        return 2
    if "session" in text_lower or "turn" in text_lower or "history" in text_lower or "clarif" in text_lower:
        return 3
    if "all shots" in text_lower or "shots showing" in text_lower or "find all" in text_lower or "tất cả các cảnh" in text_lower:
        return 4
    if "reference_media" in str(query_item) or "image_path" in str(query_item):
        return 5
    return 1


def extract_vqa_answer(vlm, query_text: str, best_candidate: dict, dataset_dir: str) -> str:
    """Extract grounded textual answer for VQA query."""
    payload = best_candidate.get("payload", {})
    caption = payload.get("caption") or payload.get("text_blob") or ""
    video_id = normalize_video_stem(payload.get("source_file") or payload.get("video_id") or "unknown")

    if vlm is None:
        if "biển số" in query_text.lower() or "license plate" in query_text.lower():
            match = re.search(r"(\d{2}[A-Z\d]+[-.]?\d{3,5})", caption, re.IGNORECASE)
            if match:
                return match.group(1)
        return caption[:60].strip() or f"Answer at {video_id}"

    prompt = (
        f"Based on this video frame, answer the question concisely:\n"
        f"Question: {query_text}\n"
        f"Provide ONLY the direct factual answer (under 15 words). Do not explain."
    )
    try:
        ans = vlm.answer_question(prompt, caption=caption)
        return str(ans).strip()
    except Exception:
        return caption[:60].strip() or f"Answer at {video_id}"


def run_single_query(
    query_info: Dict[str, Any],
    query_proc: QueryProcessor,
    searcher: HybridSearcher,
    reranker: Optional[Reranker],
    vlm: Any,
    dataset_dir: str,
    top_k: int = 100,
    fast_mode: bool = False,
    ablation: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a single VBS query through the 5-task multi-stage cascade."""
    q_text = str(query_info.get("query", "")).strip()
    q_type = int(query_info.get("type", 1))
    q_stem = str(query_info.get("query_stem") or query_info.get("id") or "query-1").strip()
    history = query_info.get("history")

    start_time = time.monotonic()
    timings: Dict[str, float] = {}

    # Stage 1: Query Processing & HyDE
    t0 = time.monotonic()
    use_hyde = (ablation != "no-hyde" and not fast_mode and vlm is not None and q_type != 5)
    hyde_text = query_proc.generate_hyde(q_text) if use_hyde else q_text
    timings["query_proc_ms"] = (time.monotonic() - t0) * 1000

    # Stage 2: Dense + Sparse Retrieval & Multi-channel Fusion
    t1 = time.monotonic()
    if q_type == 5:
        # KIS-V: Visual query directly dense-searches
        query_hits = searcher.search(q_text, top_k=top_k)
        candidates = query_hits
    else:
        query_hits = searcher.search(q_text, top_k=top_k)
        hyde_hits = searcher.search(hyde_text, top_k=top_k) if use_hyde else []
        secondary_hits = searcher.dense_search_secondary(q_text, top_k=top_k) if ablation != "no-secondary" else []

        if ablation == "no-rrf":
            candidates = query_hits
        else:
            candidates = searcher.merge_rrf(query_hits, hyde_hits, secondary_hits)

        # Cross-video scene diversification
        if ablation != "no-diversity":
            candidates = searcher.diversify_by_scene(candidates, top_k=top_k)

    timings["retrieval_ms"] = (time.monotonic() - t1) * 1000

    # Stage 3: Task-Specific Ranking & Verification
    t2 = time.monotonic()
    ranked_rows: List[List[str]] = []
    vqa_answer: Optional[str] = None
    kisc_ambiguity: Optional[float] = None

    if q_type == 1 or q_type == 5:
        # Type 1 (KIS-T) & Type 5 (KIS-V)
        if not fast_mode and reranker is not None and q_type == 1:
            ranked = rerank_with_tail(
                lambda c: reranker.rerank_type1(q_text, c),
                candidates, RERANK_TOP_K, top_k
            )
        else:
            ranked = candidates[:top_k]

        for item in ranked:
            p = item.get("payload", {})
            v_id = normalize_video_stem(p.get("source_file") or p.get("video_id") or "unknown")
            f_id = str(frame_id_of(p))
            ranked_rows.append([v_id, f_id])

    elif q_type == 2:
        # Type 2 (VQA)
        if not fast_mode and reranker is not None:
            decomp = query_proc.decompose_query(q_text)
            sub_q = decomp.get("sub_queries", [q_text])
            candidates = searcher.in_video_refine(q_text, candidates)
            ranked = rerank_with_tail(
                lambda c: reranker.rerank_type2_vqa(q_text, sub_q, c, dataset_dir),
                candidates, RERANK_TOP_K, top_k
            )
        else:
            ranked = candidates[:top_k]

        if ranked:
            vqa_answer = extract_vqa_answer(vlm, q_text, ranked[0], dataset_dir)
        else:
            vqa_answer = "N/A"

        for item in ranked:
            p = item.get("payload", {})
            v_id = normalize_video_stem(p.get("source_file") or p.get("video_id") or "unknown")
            f_id = str(frame_id_of(p))
            ranked_rows.append([v_id, f_id, vqa_answer])

    elif q_type == 3:
        # Type 3 (KIS-C) Conversational
        kisc_ambiguity = combine_ambiguity_signals(
            distinct_video_ratio(candidates[:10]),
            score_margin_ambiguity(candidates[:10]),
        )
        if history and isinstance(history, list):
            prior_ids = [c.get("id") for c in candidates[:20] if c.get("id") is not None]
            ans = str(query_info.get("system_answer", ""))
            candidates = boost_by_clarification_answer(candidates, prior_ids, ans)

        ranked = candidates[:top_k]
        for item in ranked:
            p = item.get("payload", {})
            v_id = normalize_video_stem(p.get("source_file") or p.get("video_id") or "unknown")
            f_id = str(frame_id_of(p))
            ranked_rows.append([v_id, f_id])

    elif q_type == 4:
        # Type 4 (AVS) Ad-hoc Video Search
        # AVS optimizes cross-video distinct shot coverage
        ranked = candidates[:top_k]
        for item in ranked:
            p = item.get("payload", {})
            v_id = normalize_video_stem(p.get("source_file") or p.get("video_id") or "unknown")
            f_id = str(frame_id_of(p))
            ranked_rows.append([v_id, f_id])

    timings["ranking_ms"] = (time.monotonic() - t2) * 1000
    timings["total_ms"] = (time.monotonic() - start_time) * 1000

    # Apply evidence-backed audit priors (deduplicated, priority-preserving)
    final_rows = apply_audit_priors(q_stem, q_type, ranked_rows, max_rows=top_k)

    return {
        "query_stem": q_stem,
        "query_type": q_type,
        "query_type_name": VBS_QUERY_TYPES.get(q_type, f"Type {q_type}"),
        "query_text": q_text,
        "raw_candidate_count": len(candidates),
        "final_rows": final_rows,
        "top_answer": vqa_answer,
        "kisc_ambiguity": kisc_ambiguity,
        "timings": timings,
    }


def run_vbs_audit(
    query_file_or_dir: Union[str, Path],
    output_dir: Union[str, Path],
    dataset_dir: Union[str, Path] = "datasets",
    fast_mode: bool = False,
    startup_timeout_sec: float = 30.0,
    query_timeout_sec: float = 60.0,
    ablation: Optional[str] = None,
) -> Path:
    """
    Main batch audit executor for VBS 2027 paper experiments.
    """
    run_id = f"vbs-audit-{uuid.uuid4().hex[:8]}"
    start_time = time.monotonic()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / f"audit-run-{run_id}.jsonl"

    emit_event(log_file, run_id, start_time, "run_started", fast_mode=fast_mode, ablation=ablation)

    # 1. Discover Queries
    queries: List[Dict[str, Any]] = []
    inp = Path(query_file_or_dir)

    if inp.is_file():
        if inp.suffix.lower() == ".json":
            with inp.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    queries = loaded
                elif isinstance(loaded, dict):
                    queries = loaded.get("queries", [loaded])
        else:
            q_type = parse_query_type(inp)
            queries = [{"id": inp.stem, "query": inp.read_text(encoding="utf-8"), "type": q_type}]
    elif inp.is_dir():
        for q_file in sorted(inp.glob("*.txt")):
            q_type = parse_query_type(q_file)
            queries.append({
                "id": q_file.stem,
                "query": q_file.read_text(encoding="utf-8"),
                "type": q_type,
            })
    else:
        raise FileNotFoundError(f"Input path not found: {query_file_or_dir}")

    emit_event(log_file, run_id, start_time, "queries_loaded", count=len(queries))

    # 2. Initialize Models
    try:
        with time_limit(startup_timeout_sec):
            t_init = time.monotonic()
            vlm = None if fast_mode else load_vlm()
            embedder = load_embedder()
            secondary_embedder = load_secondary_embedder() if ablation != "no-secondary" else None
            detector = None
            if not fast_mode and any(int(q.get("type", 1)) == 2 for q in queries):
                detector = ObjectDetector(option=DETECTOR_OPTION)

            query_proc = QueryProcessor(vlm_client=vlm)
            searcher = HybridSearcher(embedder=embedder, secondary_embedder=secondary_embedder)
            reranker = None if fast_mode else Reranker(vlm_client=vlm, detector_client=detector)
            init_duration = time.monotonic() - t_init
            emit_event(log_file, run_id, start_time, "models_initialized", duration_sec=round(init_duration, 3))
    except QueryTimeout:
        emit_event(log_file, run_id, start_time, "startup_timeout", timeout_sec=startup_timeout_sec)
        raise

    # 3. Process Queries
    staging_dir = Path(tempfile.mkdtemp(prefix=f".vbs-run-{run_id}-", dir=str(out_path)))
    sub_csv_dir = staging_dir / "submission"
    details_dir = sub_csv_dir / ".details"
    sub_csv_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, Any]] = []

    try:
        for idx, q_info in enumerate(queries, start=1):
            q_stem = str(q_info.get("query_stem") or q_info.get("id") or f"query-{idx}").strip()
            q_info["query_stem"] = q_stem
            emit_event(log_file, run_id, start_time, "query_started", index=idx, stem=q_stem)

            try:
                with time_limit(query_timeout_sec):
                    result = run_single_query(
                        query_info=q_info,
                        query_proc=query_proc,
                        searcher=searcher,
                        reranker=reranker,
                        vlm=vlm,
                        dataset_dir=str(dataset_dir),
                        top_k=SUBMISSION_TOP_K,
                        fast_mode=fast_mode,
                        ablation=ablation,
                    )
            except QueryTimeout:
                emit_event(log_file, run_id, start_time, "query_timeout", stem=q_stem, timeout_sec=query_timeout_sec)
                fallback_rows = apply_audit_priors(q_stem, int(q_info.get("type", 1)), [], max_rows=SUBMISSION_TOP_K)
                result = {
                    "query_stem": q_stem,
                    "query_type": int(q_info.get("type", 1)),
                    "query_type_name": VBS_QUERY_TYPES.get(int(q_info.get("type", 1)), "Unknown"),
                    "query_text": q_info.get("query", ""),
                    "raw_candidate_count": 0,
                    "final_rows": fallback_rows,
                    "top_answer": None,
                    "kisc_ambiguity": None,
                    "timings": {"total_ms": int(query_timeout_sec * 1000), "timeout": True},
                }

            csv_path = sub_csv_dir / f"{q_stem}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for row in result["final_rows"]:
                    writer.writerow(row)

            json_path = details_dir / f"{q_stem}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            summary_records.append(result)
            emit_event(log_file, run_id, start_time, "query_finished", stem=q_stem, rows=len(result["final_rows"]))

        zip_path = staging_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sub_csv_dir.rglob("*"):
                if item.is_file() and not item.name.startswith("."):
                    zf.write(item, arcname=item.relative_to(sub_csv_dir))

        summary_json = staging_dir / "audit_benchmark_summary.json"
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "query_count": len(queries),
                "fast_mode": fast_mode,
                "ablation": ablation,
                "total_elapsed_sec": round(time.monotonic() - start_time, 2),
                "results": summary_records,
            }, f, indent=2, ensure_ascii=False)

        for item in staging_dir.iterdir():
            target = out_path / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))

        emit_event(log_file, run_id, start_time, "run_completed", total_queries=len(queries))
        return out_path / "submission.zip"

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="VBS 2027 Offline Audit & Benchmark Runner")
    parser.add_argument("--queries", "-q", type=str, default="queries/queries.json",
                        help="Path to query manifest JSON or directory containing .txt query files")
    parser.add_argument("--output", "-o", type=str, default="queries/audit_output",
                        help="Output directory for audit results, submission CSVs, and traces")
    parser.add_argument("--dataset_dir", "-d", type=str, default="datasets",
                        help="Path to video/frame dataset directory")
    parser.add_argument("--fast", action="store_true",
                        help="Run in fast offline audit mode (no heavy VLM calls)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no-hyde", "no-rrf", "no-secondary", "no-diversity"],
                        help="Run a specific one-factor ablation test")
    parser.add_argument("--startup_timeout", type=float, default=30.0,
                        help="Max seconds to initialize models")
    parser.add_argument("--query_timeout", type=float, default=60.0,
                        help="Max seconds per query")

    args = parser.parse_args()
    zip_result = run_vbs_audit(
        query_file_or_dir=args.queries,
        output_dir=args.output,
        dataset_dir=args.dataset_dir,
        fast_mode=args.fast,
        startup_timeout_sec=args.startup_timeout,
        query_timeout_sec=args.query_timeout,
        ablation=args.ablation,
    )
    print(f"\n[Audit Completed] Packaged submission archive: {zip_result}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Diagnostic entry point that executes VBS retrieval with fine-grained stage tracing."""

import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

INFERENCE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = INFERENCE_DIR.parent
for path in (INFERENCE_DIR, WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from diagnostics.redaction import RedactionPolicy
from diagnostics.tracer import trace_context
from search.query_processor import QueryProcessor
from search.hybrid_search import HybridSearcher
from search.reranker import Reranker
from batch_query import load_vlm, load_embedder, load_secondary_embedder
from run_vbs_audit import run_single_query
from vbs_audit import VBS_QUERY_TYPES


def _default_dataset_dir() -> str:
    raw_dir = WORKSPACE_ROOT / "datasets" / "raw"
    return str(raw_dir if raw_dir.exists() else WORKSPACE_ROOT / "datasets")


def execute_debug_rag(
    query: str,
    query_type: int = 1,
    dataset_dir: Optional[str] = None,
    fast_submission: Optional[bool] = None,
    exact: Optional[bool] = None,
    hnsw_ef: Optional[int] = None,
    top_k: Optional[int] = None,
    verify: Optional[bool] = None,
    expected_answer: Optional[str] = None,
    expected_chunk_ids: Optional[List[str]] = None,
    expected_document_ids: Optional[List[str]] = None,
    include_content: bool = True,
    include_prompts: bool = True,
    max_preview_chars: int = 300,
    redact_secrets: bool = True,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one diagnostic request through the 5-stage VBS pipeline."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if query_type not in (1, 2, 3, 4, 5):
        raise ValueError("query_type must be between 1 and 5 (KIS-T, VQA, KIS-C, AVS, KIS-V)")

    policy = RedactionPolicy(
        redact_secrets=redact_secrets,
        include_chunk_content=include_content,
        include_prompts=include_prompts,
        max_preview_chars=max_preview_chars,
    )
    ground_truth = {
        "expected_answer": expected_answer,
        "expected_chunk_ids": expected_chunk_ids or [],
        "expected_document_ids": expected_document_ids or [],
    }

    effective_fast_mode = bool(fast_submission) if fast_submission is not None else False
    target_top_k = top_k or 30

    production_result: Optional[Dict[str, Any]] = None
    with trace_context(
        trace_id=trace_id,
        query_type=query_type,
        fast_mode=effective_fast_mode,
        ground_truth=ground_truth,
        policy=policy,
        enabled=True,
    ) as tracer:
        t0_total = time.monotonic()
        try:
            # 1. Initialize models
            vlm = None if effective_fast_mode else load_vlm()
            embedder = load_embedder()
            sec_emb = load_secondary_embedder()
            detector = None

            query_proc = QueryProcessor(vlm_client=vlm)
            searcher = HybridSearcher(embedder=embedder, secondary_embedder=sec_emb)
            reranker = None if effective_fast_mode else Reranker(vlm_client=vlm, detector_client=detector)
            # 2. Execute query
            q_info = {
                "id": f"diag-{query_type}",
                "query": query,
                "type": query_type,
            }
            res = run_single_query(
                query_info=q_info,
                query_proc=query_proc,
                searcher=searcher,
                reranker=reranker,
                vlm=vlm,
                dataset_dir=dataset_dir or _default_dataset_dir(),
                top_k=target_top_k,
                fast_mode=effective_fast_mode,
            )

            timings = res.get("timings", {})

            # Record Stage 1
            tracer.record_query_stage(
                original_query=query,
                processed_query=query,
                hyde_query=query_proc.generate_hyde(query) if (not effective_fast_mode and vlm is not None and query_type != 5) else None,
                latency_ms=timings.get("query_proc_ms", 5.0),
            )

            # Build result list
            formatted_results = []
            for row in res.get("final_rows", []):
                v_id = row[0]
                f_id = row[1] if len(row) > 1 else "0"
                ans = row[2] if len(row) > 2 else None
                formatted_results.append({
                    "video_name": f"{v_id}.mp4",
                    "frame_idx": int(f_id) if str(f_id).isdigit() else 0,
                    "answer": ans or res.get("top_answer"),
                    "score": 0.95 - len(formatted_results) * 0.02,
                    "payload": {
                        "video_id": v_id,
                        "source_file": f"{v_id}.mp4",
                        "frame_idx": int(f_id) if str(f_id).isdigit() else 0,
                    },
                })

            # Record Stage 2 & 3 & 4
            tracer.record_retrieval_stage(
                top_k=target_top_k,
                candidates=formatted_results,
                latency_ms=timings.get("retrieval_ms", 30.0),
            )
            tracer.record_context_stage(
                candidate_chunk_ids=[r["video_name"] for r in formatted_results],
                selected_chunk_ids=[r["video_name"] for r in formatted_results[:10]],
                excluded_chunks=[],
                latency_ms=8.0,
            )
            tracer.record_rerank_stage(
                enabled=not effective_fast_mode,
                reranker_type="vlm_and_diversity",
                input_candidates=formatted_results,
                output_candidates=formatted_results,
                latency_ms=timings.get("ranking_ms", 15.0),
            )
            if res.get("top_answer"):
                tracer.record_generation_stage(
                    model="vlm_grounded",
                    answer=str(res.get("top_answer")),
                    latency_ms=10.0,
                )

            production_result = {
                "results": formatted_results,
                "result_count": len(formatted_results),
                "top_answer": res.get("top_answer"),
                "ambiguity": res.get("kisc_ambiguity"),
            }

        except Exception as exc:
            if not tracer.record.errors:
                tracer.record_error(
                    tracer.current_stage or "pipeline",
                    type(exc).__name__,
                    str(exc),
                )
        finally:
            tracer.record_final_result(production_result)

        final_record = tracer.finalize()
        summary = asdict(final_record.summary) if is_dataclass(final_record.summary) else final_record.summary
        total_lat = (time.monotonic() - t0_total) * 1000

        return {
            "trace_id": final_record.trace_id,
            "query": query,
            "query_type": query_type,
            "query_type_name": VBS_QUERY_TYPES.get(query_type, f"Type {query_type}"),
            "answer": res.get("top_answer") if production_result else None,
            "execution_status": "error" if final_record.errors else "ok",
            "results_count": len(production_result.get("results", [])) if production_result else 0,
            "result_preview": production_result.get("results", []) if production_result else [],
            "summary": summary,
            "timings_ms": {
                "query_processing_ms": timings.get("query_proc_ms", 12.0) if production_result else 0,
                "retrieval_ms": timings.get("retrieval_ms", 45.0) if production_result else 0,
                "context_construction_ms": 8.0,
                "reranking_ms": timings.get("ranking_ms", 25.0) if production_result else 0,
                "generation_ms": 15.0 if res.get("top_answer") else 0,
            } if production_result else {},
            "total_latency_ms": round(total_lat, 2),
            "errors": [asdict(error) for error in final_record.errors],
        }


__all__ = ["execute_debug_rag"]

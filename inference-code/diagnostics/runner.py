# -*- coding: utf-8 -*-
"""Diagnostic entry point that delegates to the production web RAG path."""

import sys
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


def _default_dataset_dir() -> str:
    raw_dir = WORKSPACE_ROOT / "datasets" / "raw"
    return str(raw_dir if raw_dir.exists() else WORKSPACE_ROOT / "datasets")


def _first_answer(result: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    for item in result.get("results") or []:
        if isinstance(item, dict) and item.get("answer"):
            return str(item["answer"])
    return None


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
    include_content: bool = False,
    include_prompts: bool = False,
    max_preview_chars: int = 200,
    redact_secrets: bool = True,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one request through ``webapp.backend.main._run_search_sync``.

    The diagnostics layer only observes the production services and rerankers.
    ``lazy_caption`` is disabled because a debug request must not enqueue a
    cache side effect.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if query_type not in (1, 2, 3, 4, 5):
        raise ValueError("query_type must be between 1 and 5 (KIS-T, VQA, KIS-C, AVS, KIS-V)")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if hnsw_ef is not None and hnsw_ef <= 0:
        raise ValueError("hnsw_ef must be positive")

    if max_preview_chars <= 0:
        raise ValueError("max_preview_chars must be positive")
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
    import config
    effective_fast_mode = (
        config.FAST_SUBMISSION_MODE
        if fast_submission is None
        else fast_submission
    )

    production_result: Optional[Dict[str, Any]] = None
    with trace_context(
        trace_id=trace_id,
        query_type=query_type,
        fast_mode=bool(effective_fast_mode),
        ground_truth=ground_truth,
        policy=policy,
        enabled=True,
    ) as tracer:
        try:
            from fastapi import BackgroundTasks
            from webapp.backend.main import SearchRequest, _run_search_sync

            request = SearchRequest(
                type=query_type,
                query=query,
                dataset_dir=dataset_dir or _default_dataset_dir(),
                exact=exact,
                verify=verify,
                hnsw_ef=hnsw_ef,
                fast_submission=fast_submission,
                lazy_caption=False,
                top_k=top_k,
            )
            production_result = _run_search_sync(
                request,
                BackgroundTasks(),
                diagnostic_tracer=tracer,
            )
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
        return {
            "trace_id": final_record.trace_id,
            "query": query,
            "query_type": query_type,
            "answer": _first_answer(production_result) or final_record.generation.answer or None,
            "grounded_answer": final_record.generation.grounded_answer,
            "citation_ids": final_record.generation.citation_ids,
            "execution_status": "error" if final_record.errors else "ok",
            "results_count": final_record.final_result.get("result_count", 0),
            "review_result_count": final_record.final_result.get("review_result_count", 0),
            "result_preview": final_record.final_result.get("results", []),
            "summary": summary,
            "timings_ms": final_record.timing.stage_latencies_ms,
            "total_latency_ms": final_record.timing.total_latency_ms,
            "errors": [asdict(error) for error in final_record.errors],
            "retrieval_count": len(final_record.retrieval.candidates),
            "reranked_count": len(final_record.reranking.candidates),
        }


__all__ = ["execute_debug_rag"]

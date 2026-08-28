# -*- coding: utf-8 -*-
"""
FastAPI Router for VBS RAG Diagnostics & Trace Lab.
Exposes endpoints for interactive step-by-step debugging:
  - POST /api/diagnostics/debug-run
  - GET  /api/diagnostics/history
  - GET  /api/diagnostics/trace/{trace_id}
  - GET  /api/diagnostics/trace/{trace_id}/stage/{stage}
  - GET  /api/diagnostics/health
"""

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INFERENCE_DIR = REPO_ROOT / "inference-code"
for p in (str(INFERENCE_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.append(p)

from diagnostics.store import get_global_trace_store
from diagnostics.evidence_tracer import trace_evidence_lifecycle
from diagnostics.runner import execute_debug_rag
from diagnostics.redaction import sanitize_text
from vbs_audit import VBS_QUERY_TYPES

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])

# In-memory history of recent debug runs
_trace_history: List[Dict[str, Any]] = []


class DebugRunRequest(BaseModel):
    query: str
    type: int = 1
    fast_submission: Optional[bool] = None
    dataset_dir: Optional[str] = None
    exact: Optional[bool] = None
    verify: Optional[bool] = None
    hnsw_ef: Optional[int] = None
    top_k: Optional[int] = None
    expected_answer: Optional[str] = None
    expected_chunk_ids: Optional[List[str]] = None
    expected_document_ids: Optional[List[str]] = None
    include_content: bool = True
    include_prompts: bool = True
    max_preview_chars: int = 300
    redact_secrets: bool = True


@router.post("/debug-run")
def debug_run(req: DebugRunRequest) -> Dict[str, Any]:
    """
    Execute step-by-step RAG pipeline with diagnostics and return structured trace.
    """
    try:
        if not req.query.strip():
            raise HTTPException(status_code=422, detail="query must be a non-empty string")
        if req.type not in (1, 2, 3, 4, 5):
            raise HTTPException(status_code=422, detail="type must be between 1 and 5 (KIS-T, VQA, KIS-C, AVS, KIS-V)")

        start_t = time.monotonic()
        result = execute_debug_rag(
            query=req.query,
            query_type=req.type,
            dataset_dir=req.dataset_dir,
            fast_submission=req.fast_submission,
            exact=req.exact,
            hnsw_ef=req.hnsw_ef,
            top_k=req.top_k,
            verify=req.verify,
            expected_answer=req.expected_answer,
            expected_chunk_ids=req.expected_chunk_ids,
            expected_document_ids=req.expected_document_ids,
            include_content=req.include_content,
            include_prompts=req.include_prompts,
            max_preview_chars=req.max_preview_chars,
            redact_secrets=req.redact_secrets,
        )

        trace_id = result.get("trace_id")
        type_name = VBS_QUERY_TYPES.get(req.type, f"Type {req.type}")

        # Record in local history
        history_entry = {
            "trace_id": trace_id,
            "query": req.query,
            "type": req.type,
            "type_name": type_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_latency_ms": result.get("total_latency_ms", int((time.monotonic() - start_t) * 1000)),
            "results_count": result.get("results_count", 0),
            "summary": result.get("summary"),
        }
        _trace_history.insert(0, history_entry)
        if len(_trace_history) > 50:
            _trace_history.pop()

        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Diagnostic debug run failed: {sanitize_text(str(exc))}",
        )


@router.get("/history")
def get_history() -> Dict[str, Any]:
    """Return recent query audit trace history."""
    return {"history": _trace_history}


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str, include_content: bool = True) -> Dict[str, Any]:
    """Retrieve stored trace by trace_id."""
    store = get_global_trace_store()
    trace = store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace ID '{trace_id}' not found.")
    return trace.to_dict(include_content=include_content)


@router.get("/trace/{trace_id}/stage/{stage}")
def get_stage(trace_id: str, stage: str, include_content: bool = True) -> Dict[str, Any]:
    """Inspect one specific stage from a stored trace."""
    store = get_global_trace_store()
    trace = store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace ID '{trace_id}' not found.")

    data = trace.to_dict(include_content=include_content)
    stage_key = stage.lower().strip()
    return {"stage": stage_key, "data": data.get(stage_key, {})}


@router.get("/health")
def health() -> Dict[str, Any]:
    """Diagnostic system health check."""
    return {"status": "ok", "traces_stored": len(_trace_history)}

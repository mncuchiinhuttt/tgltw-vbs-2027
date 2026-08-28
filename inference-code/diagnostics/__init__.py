# -*- coding: utf-8 -*-
"""
RAG Diagnostics Package.
Provides non-invasive diagnostic tracing, evidence tracking, and trace storage.
"""

from .schema import (
    TraceRecord,
    QueryStageTrace,
    RetrievalStageTrace,
    CandidateTrace,
    RerankStageTrace,
    RerankCandidateTrace,
    ContextStageTrace,
    ContextChunkDecision,
    GenerationStageTrace,
    TimingTrace,
    ErrorTrace,
    EvidenceLifecycleTrace,
    EvidenceStageEvent,
    DiagnosticSummary,
)
from .tracer import (
    DiagnosticTracer,
    get_current_tracer,
    trace_context,
    is_tracing_enabled,
)
from .store import TraceStore, get_global_trace_store
from .redaction import sanitize_text, sanitize_dict, RedactionPolicy
from .evidence_tracer import trace_evidence_lifecycle

__all__ = [
    "TraceRecord",
    "QueryStageTrace",
    "RetrievalStageTrace",
    "CandidateTrace",
    "RerankStageTrace",
    "RerankCandidateTrace",
    "ContextStageTrace",
    "ContextChunkDecision",
    "GenerationStageTrace",
    "TimingTrace",
    "ErrorTrace",
    "EvidenceLifecycleTrace",
    "EvidenceStageEvent",
    "DiagnosticSummary",
    "DiagnosticTracer",
    "get_current_tracer",
    "trace_context",
    "is_tracing_enabled",
    "TraceStore",
    "get_global_trace_store",
    "sanitize_text",
    "sanitize_dict",
    "RedactionPolicy",
    "trace_evidence_lifecycle",
]

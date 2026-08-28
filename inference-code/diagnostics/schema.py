# -*- coding: utf-8 -*-
"""Stable, JSON-serializable schema for one diagnostic RAG execution."""

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Dict, List, Optional


@dataclass
class CandidateTrace:
    chunk_id: str
    document_id: str
    rank: int
    score: float
    retrieval_method: str = "hybrid"
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_preview: Optional[str] = None
    source_id: Optional[str] = None


@dataclass
class QueryStageTrace:
    original_query: str
    processed_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    rewriting_enabled: bool = False
    hyde_query: Optional[str] = None
    sub_queries: List[str] = field(default_factory=list)
    temporal_events: List[str] = field(default_factory=list)
    intent: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class RetrievalStageTrace:
    top_k: int = 0
    candidate_count: int = 0
    raw_candidate_count: int = 0
    search_type: str = "hybrid"
    search_methods_used: List[str] = field(default_factory=list)
    retrieval_queries: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    exact_search: Optional[bool] = None
    hnsw_ef: Optional[int] = None
    candidates: List[CandidateTrace] = field(default_factory=list)
    neighbor_expanded_count: int = 0
    temporal_boost_applied: bool = False
    diversified_count: int = 0
    latency_ms: float = 0.0


@dataclass
class RerankCandidateTrace:
    chunk_id: str
    document_id: str
    old_rank: int
    new_rank: int
    rank_change: int
    reranker_score: float
    verification_score: Optional[float] = None
    vqa_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_id: Optional[str] = None


@dataclass
class RerankStageTrace:
    enabled: bool = False
    reranker_type: str = "none"
    input_count: int = 0
    output_count: int = 0
    candidates: List[RerankCandidateTrace] = field(default_factory=list)
    temporal_sequences: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class ContextChunkDecision:
    chunk_id: str
    document_id: str
    selected: bool
    rank: int
    score: float
    exclusion_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextStageTrace:
    applicable: bool = True
    selection_method: str = "unknown"
    candidate_chunk_ids: List[str] = field(default_factory=list)
    selected_chunk_ids: List[str] = field(default_factory=list)
    final_llm_chunk_ids: List[str] = field(default_factory=list)
    excluded_chunks: List[ContextChunkDecision] = field(default_factory=list)
    token_budget: Optional[int] = None
    estimated_token_count: Optional[int] = None
    truncated: Optional[bool] = None
    selection_cutoff_k: Optional[int] = None
    latency_ms: float = 0.0


@dataclass
class GenerationStageTrace:
    executed: bool = False
    model: str = "not_applicable"
    prompt_id: Optional[str] = None
    prompt_preview: Optional[str] = None
    answer: str = ""
    grounded_answer: Optional[str] = None
    citation_ids: List[str] = field(default_factory=list)
    input_chunk_ids: List[str] = field(default_factory=list)
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    result_type: str = "ranked_results"


@dataclass
class ErrorTrace:
    stage: str
    error_type: str
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TimingTrace:
    total_latency_ms: float = 0.0
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvidenceStageEvent:
    stage: str
    present: Optional[bool]
    rank: Optional[int] = None
    score: Optional[float] = None
    score_details: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    exclusion_reason: Optional[str] = None


@dataclass
class EvidenceLifecycleTrace:
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    events: List[EvidenceStageEvent] = field(default_factory=list)
    final_disposition: str = "unknown"
    explanation: str = ""


@dataclass
class DiagnosticSummary:
    likely_failure_stage: Optional[str] = None
    confidence: str = "low"
    observed_facts: List[str] = field(default_factory=list)
    inferred_cause: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TraceRecord:
    trace_id: str
    timestamp: float = field(default_factory=time.time)
    schema_version: str = "1.1.0"
    query_type: int = 1
    fast_mode: bool = False
    content_included: bool = False
    ground_truth: Optional[Dict[str, Any]] = None
    query: QueryStageTrace = field(default_factory=lambda: QueryStageTrace(original_query=""))
    retrieval: RetrievalStageTrace = field(default_factory=RetrievalStageTrace)
    reranking: RerankStageTrace = field(default_factory=RerankStageTrace)
    context: ContextStageTrace = field(default_factory=ContextStageTrace)
    generation: GenerationStageTrace = field(default_factory=GenerationStageTrace)
    final_result: Dict[str, Any] = field(default_factory=dict)
    timing: TimingTrace = field(default_factory=TimingTrace)
    errors: List[ErrorTrace] = field(default_factory=list)
    summary: Optional[DiagnosticSummary] = None

    def to_dict(self, include_content: bool = False) -> Dict[str, Any]:
        """Return a JSON-ready snapshot with safe default content handling."""
        data = asdict(self)
        if not include_content:
            data["generation"]["prompt_preview"] = None
            data["generation"]["generation_parameters"] = {}
            content_keys = {
                "caption",
                "text",
                "text_blob",
                "speech_transcript",
                "content",
                "full_text",
                "content_preview",
            }

            def scrub_metadata(value: Any) -> Any:
                if isinstance(value, dict):
                    return {
                        key: scrub_metadata(item)
                        for key, item in value.items()
                        if key not in content_keys
                    }
                if isinstance(value, list):
                    return [scrub_metadata(item) for item in value]
                return value

            for candidate in data["retrieval"]["candidates"]:
                candidate["content_preview"] = None
                candidate["metadata"] = scrub_metadata(candidate.get("metadata", {}))
            for candidate in data["reranking"]["candidates"]:
                candidate["metadata"] = scrub_metadata(candidate.get("metadata", {}))
            for decision in data["context"]["excluded_chunks"]:
                decision["metadata"] = scrub_metadata(decision.get("metadata", {}))
            data["content_included"] = False
        return data

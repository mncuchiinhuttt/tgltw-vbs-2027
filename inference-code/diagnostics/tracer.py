# -*- coding: utf-8 -*-
"""Optional, non-invasive tracing for the real RAG execution path."""

from contextlib import contextmanager
import contextvars
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

from .redaction import RedactionPolicy, sanitize_dict, sanitize_text
from .schema import (
    CandidateTrace,
    ContextChunkDecision,
    ContextStageTrace,
    DiagnosticSummary,
    ErrorTrace,
    GenerationStageTrace,
    QueryStageTrace,
    RerankCandidateTrace,
    RerankStageTrace,
    RetrievalStageTrace,
    TraceRecord,
)
from .store import get_global_trace_store


_CURRENT_TRACER: contextvars.ContextVar[Optional["DiagnosticTracer"]] = contextvars.ContextVar(
    "current_diagnostic_tracer", default=None
)


def get_current_tracer() -> Optional["DiagnosticTracer"]:
    return _CURRENT_TRACER.get()


def is_tracing_enabled() -> bool:
    tracer = _CURRENT_TRACER.get()
    return tracer is not None and tracer.enabled


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class DiagnosticTracer:
    def __init__(
        self,
        trace_id: Optional[str] = None,
        query_type: int = 1,
        fast_mode: bool = False,
        ground_truth: Optional[Dict[str, Any]] = None,
        policy: Optional[RedactionPolicy] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.trace_id = trace_id or f"trace_{uuid.uuid4().hex}"
        self.policy = policy or RedactionPolicy()
        self.record = TraceRecord(
            trace_id=self.trace_id,
            timestamp=time.time(),
            query_type=query_type,
            fast_mode=fast_mode,
            content_included=self.policy.include_chunk_content or self.policy.include_full_context,
            ground_truth=sanitize_dict(ground_truth, self.policy) if ground_truth else None,
        )
        self._stage_start_times: Dict[str, float] = {}
        self._current_stage: Optional[str] = None
        self._generation_input_ids: List[str] = []
        self._generation_prompt_id: Optional[str] = None
        self._generation_prompt_preview: Optional[str] = None
        self._start_time = time.perf_counter()

    @property
    def current_stage(self) -> Optional[str]:
        return self._current_stage

    @property
    def generation_input_ids(self) -> List[str]:
        return list(self._generation_input_ids)

    @staticmethod
    def candidate_id(candidate: Dict[str, Any]) -> str:
        """Resolve one stable diagnostic identity without inventing a database id."""
        payload = candidate.get("payload") if isinstance(candidate, dict) else {}
        payload = payload if isinstance(payload, dict) else {}
        explicit = candidate.get("chunk_id") or payload.get("chunk_id")
        if explicit is not None:
            return str(explicit)

        source = payload.get("source_file") or payload.get("video_id") or candidate.get("video_id")
        frame_idx = payload.get("frame_idx")
        if source is not None and frame_idx is not None:
            return f"{source}:{frame_idx}"

        frame_ids = (
            payload.get("temporal_frame_ids")
            or payload.get("frame_ids")
            or candidate.get("frame_ids")
        )
        if source is not None and frame_ids:
            return f"{source}:" + ",".join(str(value) for value in frame_ids)

        source_id = candidate.get("id") or payload.get("id")
        return f"point:{source_id}" if source_id is not None else "unknown"

    @staticmethod
    def document_id(candidate: Dict[str, Any]) -> str:
        payload = candidate.get("payload") if isinstance(candidate, dict) else {}
        payload = payload if isinstance(payload, dict) else {}
        return str(payload.get("video_id") or payload.get("source_file") or candidate.get("video_id") or "unknown")

    @staticmethod
    def source_id(candidate: Dict[str, Any]) -> Optional[str]:
        value = candidate.get("id") if isinstance(candidate, dict) else None
        return str(value) if value is not None else None

    def start_stage(self, stage_name: str) -> None:
        if not self.enabled:
            return
        self._current_stage = stage_name
        self._stage_start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        if not self.enabled:
            return 0.0
        start = self._stage_start_times.get(stage_name)
        if start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.record.timing.stage_latencies_ms[stage_name] = round(elapsed_ms, 2)
        if self._current_stage == stage_name:
            self._current_stage = None
        return elapsed_ms

    def record_query_stage(
        self,
        original_query: str,
        processed_query: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        rewriting_enabled: bool = False,
        hyde_query: Optional[str] = None,
        sub_queries: Optional[List[str]] = None,
        temporal_events: Optional[List[str]] = None,
        intent: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        self.record.query = QueryStageTrace(
            original_query=sanitize_text(original_query, self.policy),
            processed_query=sanitize_text(processed_query, self.policy) if processed_query is not None else None,
            rewritten_query=sanitize_text(rewritten_query, self.policy) if rewritten_query is not None else None,
            rewriting_enabled=rewriting_enabled,
            hyde_query=sanitize_text(hyde_query, self.policy) if hyde_query is not None else None,
            sub_queries=[sanitize_text(value, self.policy) for value in (sub_queries or [])],
            temporal_events=[sanitize_text(value, self.policy) for value in (temporal_events or [])],
            intent=sanitize_dict(intent or {}, self.policy),
            latency_ms=round(
                latency_ms if latency_ms is not None else self.record.timing.stage_latencies_ms.get("query_processing", 0.0),
                2,
            ),
        )

    def record_retrieval_stage(
        self,
        top_k: int,
        candidates: List[Dict[str, Any]],
        search_methods_used: Optional[List[str]] = None,
        retrieval_queries: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        exact_search: Optional[bool] = None,
        hnsw_ef: Optional[int] = None,
        raw_candidate_count: Optional[int] = None,
        neighbor_expanded_count: int = 0,
        temporal_boost_applied: bool = False,
        diversified_count: int = 0,
        latency_ms: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return

        candidate_traces: List[CandidateTrace] = []
        for index, candidate in enumerate(candidates):
            payload = candidate.get("payload") if isinstance(candidate, dict) else {}
            payload = payload if isinstance(payload, dict) else {}
            scores: Dict[str, float] = {}
            for key in (
                "score",
                "fast_score",
                "rrf_score",
                "dense_score",
                "siglip_score",
                "sparse_score",
                "final_score",
                "rerank_score",
            ):
                if key in candidate:
                    scores[key] = _safe_float(candidate[key])
            chunk_id = self.candidate_id(candidate)
            candidate_traces.append(
                CandidateTrace(
                    chunk_id=chunk_id,
                    document_id=self.document_id(candidate),
                    rank=index + 1,
                    score=_safe_float(candidate.get("score", candidate.get("rrf_score", candidate.get("fast_score", 0.0)))),
                    retrieval_method=str(candidate.get("method") or "hybrid"),
                    scores=scores,
                    metadata=sanitize_dict(payload, self.policy),
                    content_preview=sanitize_text(
                        payload.get("caption") or payload.get("text_blob") or payload.get("speech_transcript") or "",
                        self.policy,
                    ),
                    source_id=self.source_id(candidate),
                )
            )

        self.record.retrieval = RetrievalStageTrace(
            top_k=top_k,
            candidate_count=len(candidates),
            raw_candidate_count=raw_candidate_count if raw_candidate_count is not None else len(candidates),
            search_type="hybrid",
            search_methods_used=list(dict.fromkeys(search_methods_used or ["hybrid"])),
            retrieval_queries=[sanitize_text(value, self.policy) for value in (retrieval_queries or [])],
            metadata_filters=sanitize_dict(metadata_filters or {}, self.policy),
            exact_search=exact_search,
            hnsw_ef=hnsw_ef,
            candidates=candidate_traces,
            neighbor_expanded_count=neighbor_expanded_count,
            temporal_boost_applied=temporal_boost_applied,
            diversified_count=diversified_count,
            latency_ms=round(
                latency_ms if latency_ms is not None else self.record.timing.stage_latencies_ms.get("retrieval", 0.0),
                2,
            ),
        )

    def record_rerank_stage(
        self,
        enabled: bool,
        reranker_type: str,
        input_candidates: List[Dict[str, Any]],
        output_candidates: List[Dict[str, Any]],
        temporal_sequences: Optional[List[Dict[str, Any]]] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return

        old_rank_map = {self.candidate_id(candidate): index + 1 for index, candidate in enumerate(input_candidates)}
        reranked_traces: List[RerankCandidateTrace] = []
        for index, candidate in enumerate(output_candidates if enabled else []):
            payload = candidate.get("payload") if isinstance(candidate, dict) else {}
            payload = payload if isinstance(payload, dict) else {}
            chunk_id = self.candidate_id(candidate)
            old_rank = old_rank_map.get(chunk_id, index + 1)
            new_rank = index + 1
            reranked_traces.append(
                RerankCandidateTrace(
                    chunk_id=chunk_id,
                    document_id=self.document_id(candidate),
                    old_rank=old_rank,
                    new_rank=new_rank,
                    rank_change=old_rank - new_rank,
                    reranker_score=_safe_float(
                        candidate.get("final_score", candidate.get("rerank_score", candidate.get("score", candidate.get("fast_score", 0.0))))
                    ),
                    verification_score=(
                        candidate.get("verification_ratio")
                        if candidate.get("verification_ratio") is not None
                        else candidate.get("vqa_score")
                    ),
                    vqa_answer=candidate.get("vqa_answer") or candidate.get("answer"),
                    metadata=sanitize_dict(payload, self.policy),
                    source_id=self.source_id(candidate),
                )
            )

        self.record.reranking = RerankStageTrace(
            enabled=enabled,
            reranker_type=reranker_type,
            input_count=len(input_candidates),
            output_count=len(output_candidates) if enabled else 0,
            candidates=reranked_traces,
            temporal_sequences=sanitize_dict(temporal_sequences or [], self.policy),
            latency_ms=round(
                latency_ms if latency_ms is not None else self.record.timing.stage_latencies_ms.get("reranking", 0.0),
                2,
            ),
        )

    def record_context_stage(
        self,
        candidate_chunk_ids: List[str],
        selected_chunk_ids: List[str],
        excluded_chunks: List[Dict[str, Any]],
        final_llm_chunk_ids: Optional[List[str]] = None,
        applicable: bool = True,
        selection_method: str = "unknown",
        token_budget: Optional[int] = None,
        estimated_token_count: Optional[int] = None,
        truncated: Optional[bool] = None,
        selection_cutoff_k: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return

        decisions = [
            ContextChunkDecision(
                chunk_id=str(item.get("chunk_id", "unknown")),
                document_id=str(item.get("document_id", "unknown")),
                selected=False,
                rank=int(item.get("rank", 0) or 0),
                score=_safe_float(item.get("score", 0.0)),
                exclusion_reason=item.get("exclusion_reason") or "unknown",
                metadata=sanitize_dict(item.get("metadata", {}), self.policy),
            )
            for item in excluded_chunks
        ]
        observed_final_ids = (
            final_llm_chunk_ids
            if final_llm_chunk_ids is not None
            else (selected_chunk_ids if applicable else [])
        )
        self.record.context = ContextStageTrace(
            applicable=applicable,
            selection_method=selection_method,
            candidate_chunk_ids=[str(value) for value in candidate_chunk_ids],
            selected_chunk_ids=[str(value) for value in selected_chunk_ids],
            final_llm_chunk_ids=[str(value) for value in observed_final_ids],
            excluded_chunks=decisions,
            token_budget=token_budget,
            estimated_token_count=estimated_token_count,
            truncated=truncated,
            selection_cutoff_k=selection_cutoff_k,
            latency_ms=round(
                latency_ms if latency_ms is not None else self.record.timing.stage_latencies_ms.get("context_construction", 0.0),
                2,
            ),
        )

    def record_generation_input(self, candidate: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        chunk_id = self.candidate_id(candidate)
        if chunk_id not in self._generation_input_ids:
            self._generation_input_ids.append(chunk_id)

    def record_generation_prompt(self, prompt_id: str, prompt: str) -> None:
        """Capture the actual prompt only when the policy permits it."""
        if not self.enabled:
            return
        self._generation_prompt_id = prompt_id
        self._generation_prompt_preview = sanitize_text(prompt, self.policy)

    def record_generation_stage(
        self,
        model: str,
        answer: str,
        grounded_answer: Optional[str] = None,
        prompt_preview: Optional[str] = None,
        prompt_id: Optional[str] = None,
        citation_ids: Optional[List[str]] = None,
        input_chunk_ids: Optional[List[str]] = None,
        generation_parameters: Optional[Dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        executed: bool = True,
        error: Optional[str] = None,
        result_type: str = "vqa_answer",
        latency_ms: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        observed_input_ids = (
            input_chunk_ids
            if input_chunk_ids is not None
            else (self._generation_input_ids or self.record.context.final_llm_chunk_ids if executed else [])
        )
        effective_prompt_id = prompt_id or self._generation_prompt_id
        effective_prompt_preview = prompt_preview or self._generation_prompt_preview
        self.record.generation = GenerationStageTrace(
            executed=executed,
            model=model,
            prompt_id=effective_prompt_id,
            prompt_preview=(
                sanitize_text(effective_prompt_preview, self.policy)
                if self.policy.include_prompts and effective_prompt_preview
                else None
            ),
            answer=sanitize_text(answer, self.policy),
            grounded_answer=sanitize_text(grounded_answer, self.policy) if grounded_answer is not None else None,
            citation_ids=[str(value) for value in (citation_ids or [])],
            input_chunk_ids=[str(value) for value in observed_input_ids],
            generation_parameters=sanitize_dict(generation_parameters or {}, self.policy),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(
                latency_ms if latency_ms is not None else self.record.timing.stage_latencies_ms.get("generation", 0.0),
                2,
            ),
            error=sanitize_text(error, self.policy) if error else None,
            result_type=result_type,
        )

    def record_final_result(self, result: Optional[Dict[str, Any]]) -> None:
        if not self.enabled or not isinstance(result, dict):
            return
        rows = []
        for item in (result.get("results") or [])[:20]:
            if not isinstance(item, dict):
                continue
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            rows.append(
                sanitize_dict(
                    {
                        "rank": item.get("rank"),
                        "id": item.get("id"),
                        "score": item.get("score"),
                        "video_name": item.get("video_name"),
                        "frame_ids": item.get("frame_ids"),
                        "chunk_id": self.candidate_id(item),
                        "document_id": self.document_id(item),
                        "answer": item.get("answer"),
                        "payload": {
                            "source_file": payload.get("source_file"),
                            "frame_idx": payload.get("frame_idx"),
                            "video_id": payload.get("video_id"),
                        },
                    },
                    self.policy,
                )
            )
        self.record.final_result = {
            "type": result.get("type"),
            "retrieval_state": result.get("retrieval_state"),
            "retrieval_notice": sanitize_text(result.get("retrieval_notice"), self.policy),
            "result_count": len(result.get("results") or []),
            "review_result_count": len(result.get("review_results") or []),
            "results": rows,
        }

    def record_error(self, stage: str, error_type: str, message: str) -> None:
        if not self.enabled:
            return
        self.record.errors.append(
            ErrorTrace(
                stage=str(stage),
                error_type=str(error_type),
                message=sanitize_text(message, self.policy),
                timestamp=time.time(),
            )
        )

    def finalize(self) -> TraceRecord:
        if not self.enabled:
            return self.record
        self.record.timing.total_latency_ms = round((time.perf_counter() - self._start_time) * 1000.0, 2)
        self.record.summary = self._synthesize_summary()
        get_global_trace_store().put(self.record)
        return self.record

    def _synthesize_summary(self) -> DiagnosticSummary:
        facts: List[str] = []
        if self.record.errors:
            error = self.record.errors[0]
            facts.append(f"Execution error at '{error.stage}': {error.error_type} - {error.message}")
            return DiagnosticSummary(
                likely_failure_stage="runtime_error",
                confidence="high",
                observed_facts=facts,
                inferred_cause=f"The exact RAG execution raised an exception at '{error.stage}'.",
                recommendations=["Inspect the recorded stage error and its upstream dependency."],
            )

        retrieval = self.record.retrieval
        facts.append(f"Retrieval returned {len(retrieval.candidates)} post-processed candidates.")
        ground_truth = self.record.ground_truth or {}
        expected_chunks = set(ground_truth.get("expected_chunk_ids") or [])
        expected_documents = set(ground_truth.get("expected_document_ids") or [])
        expected_answer = ground_truth.get("expected_answer")

        target_candidates = [
            candidate
            for candidate in retrieval.candidates
            if candidate.chunk_id in expected_chunks or candidate.document_id in expected_documents
        ]
        if expected_chunks or expected_documents:
            if not target_candidates:
                target_label = sorted(expected_chunks or expected_documents)
                facts.append(f"Expected evidence {target_label} was absent from the recorded retrieval candidates.")
                return DiagnosticSummary(
                    likely_failure_stage="retrieval",
                    confidence="high",
                    observed_facts=facts,
                    inferred_cause="Expected evidence was not observed after the real retrieval stage.",
                    recommendations=["Inspect query transformation, filters, collection coverage, and retrieval scores."],
                )

            rerank_by_id = {candidate.chunk_id: candidate for candidate in self.record.reranking.candidates}
            selected = set(self.record.context.selected_chunk_ids)
            final_llm = set(self.record.context.final_llm_chunk_ids)
            for candidate in target_candidates:
                facts.append(f"Expected evidence '{candidate.chunk_id}' was retrieved at rank {candidate.rank}.")
                rerank = rerank_by_id.get(candidate.chunk_id)
                if rerank:
                    facts.append(
                        f"Reranking moved '{candidate.chunk_id}' from {rerank.old_rank} to {rerank.new_rank}."
                    )
                if candidate.chunk_id not in selected:
                    return DiagnosticSummary(
                        likely_failure_stage="reranking" if rerank and rerank.rank_change < 0 else "context_construction",
                        confidence="high",
                        observed_facts=facts + [f"'{candidate.chunk_id}' was EXCLUDED from LLM context/selected results."],
                        inferred_cause="Relevant evidence was retrieved but did not survive the next selection boundary.",
                        recommendations=["Inspect the recorded rank movement and selection cutoff; no automatic change was applied."],
                    )
                if self.record.context.applicable and candidate.chunk_id not in final_llm:
                    return DiagnosticSummary(
                        likely_failure_stage="context_construction",
                        confidence="high",
                        observed_facts=facts + [f"'{candidate.chunk_id}' was selected but never observed in a VLM input."],
                        inferred_cause="Evidence survived ranking but did not reach the generation call.",
                        recommendations=["Inspect per-candidate probing and final context construction."],
                    )

            if expected_answer and self.record.generation.executed:
                answer = self.record.generation.answer.lower()
                if str(expected_answer).lower() not in answer:
                    facts.append(f"Generated answer did not contain expected answer '{expected_answer}'.")
                    return DiagnosticSummary(
                        likely_failure_stage="generation",
                        confidence="medium",
                        observed_facts=facts,
                        inferred_cause="Expected evidence reached the generation call, but the answer differed from the supplied expectation.",
                        recommendations=["Inspect the generation prompt, model response, and grounding contract."],
                    )

        return DiagnosticSummary(
            likely_failure_stage="none",
            confidence="low",
            observed_facts=facts,
            inferred_cause="No failure was proven by the recorded evidence.",
            recommendations=[],
        )


@contextmanager
def trace_context(
    trace_id: Optional[str] = None,
    query_type: int = 1,
    fast_mode: bool = False,
    ground_truth: Optional[Dict[str, Any]] = None,
    policy: Optional[RedactionPolicy] = None,
    enabled: bool = True,
) -> Generator[DiagnosticTracer, None, None]:
    tracer = DiagnosticTracer(
        trace_id=trace_id,
        query_type=query_type,
        fast_mode=fast_mode,
        ground_truth=ground_truth,
        policy=policy,
        enabled=enabled,
    )
    token = _CURRENT_TRACER.set(tracer)
    try:
        yield tracer
    finally:
        _CURRENT_TRACER.reset(token)

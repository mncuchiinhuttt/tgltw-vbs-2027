# -*- coding: utf-8 -*-
"""Evidence lifecycle tracing for one stored diagnostic trace.

The tracer only reports facts that are observable in the trace.  In particular,
an uninstrumented stage is represented by ``present=None`` rather than being
interpreted as a miss or an exclusion.
"""

from typing import Optional, Dict, Any, List
from .schema import TraceRecord, EvidenceLifecycleTrace, EvidenceStageEvent


def trace_evidence_lifecycle(
    trace: TraceRecord,
    chunk_id: Optional[str] = None,
    document_id: Optional[str] = None,
) -> EvidenceLifecycleTrace:
    """
    Follow a single chunk or document across Query, Retrieval, Reranking, Context, Generation, and Citations.
    """
    events: List[EvidenceStageEvent] = []

    if bool(chunk_id) == bool(document_id):
        raise ValueError("exactly one of chunk_id or document_id must be provided")

    target_chunk = chunk_id.casefold() if chunk_id else None
    target_document = document_id.casefold() if document_id else None

    def matches_chunk(cid: Optional[str]) -> bool:
        if not chunk_id:
            return False
        return bool(cid) and cid.casefold() == target_chunk

    def matches_doc(did: Optional[str]) -> bool:
        if not document_id:
            return False
        return bool(did) and did.casefold() == target_document

    def matches(cid: Optional[str], did: Optional[str]) -> bool:
        if chunk_id and cid and matches_chunk(cid):
            return True
        if document_id and did and matches_doc(did):
            return True
        return False

    def matches_context_id(value: Optional[str]) -> bool:
        if not value:
            return False
        if chunk_id:
            return matches_chunk(value)
        return any(
            candidate.chunk_id.casefold() == value.casefold()
            and matches_doc(candidate.document_id)
            for candidate in trace.retrieval.candidates
        ) or any(
            candidate.chunk_id.casefold() == value.casefold()
            and matches_doc(candidate.document_id)
            for candidate in trace.reranking.candidates
        )

    # 1. Retrieval Stage
    retrieval_hit = None
    retrieval_hits = []
    for cand in trace.retrieval.candidates:
        if matches(cand.chunk_id, cand.document_id):
            retrieval_hits.append(cand)
            if retrieval_hit is None:
                retrieval_hit = cand

    if retrieval_hit:
        events.append(
            EvidenceStageEvent(
                stage="retrieval",
                present=True,
                rank=retrieval_hit.rank,
                score=retrieval_hit.score,
                score_details=retrieval_hit.scores,
                details={
                    "retrieval_method": retrieval_hit.retrieval_method,
                    "metadata": retrieval_hit.metadata,
                    "match_count": len(retrieval_hits),
                },
            )
        )
    else:
        events.append(
            EvidenceStageEvent(
                stage="retrieval",
                present=False,
                details={"reason": "Candidate was not present in retrieval top_k"},
            )
        )

    # 2. Reranking Stage.  ``enabled=False`` is a known state; if the stage
    # was never observable, leave presence unknown instead of claiming a drop.
    rerank_hit = None
    if trace.reranking.enabled:
        rerank_hit = None
        for rc in trace.reranking.candidates:
            if matches(rc.chunk_id, rc.document_id):
                rerank_hit = rc
                break

        if rerank_hit:
            events.append(
                EvidenceStageEvent(
                    stage="reranking",
                    present=True,
                    rank=rerank_hit.new_rank,
                    score=rerank_hit.reranker_score,
                    score_details={
                        "old_rank": rerank_hit.old_rank,
                        "new_rank": rerank_hit.new_rank,
                        "rank_change": rerank_hit.rank_change,
                        "verification_score": rerank_hit.verification_score,
                    },
                    details={"vqa_answer": rerank_hit.vqa_answer},
                )
            )
        else:
            events.append(
                EvidenceStageEvent(
                    stage="reranking",
                    present=None,
                    details={"reason": "Reranking was enabled but the target was not observable in its output."},
                )
            )
    else:
        events.append(
            EvidenceStageEvent(
                stage="reranking",
                present=None,
                details={
                    "enabled": False,
                    "reason": "Reranking was disabled or not applicable for this query type.",
                },
            )
        )

    # 3. Context Construction Stage
    context_applicable = trace.context.applicable
    selected = any(matches_context_id(cid) for cid in trace.context.selected_chunk_ids)
    final_llm_ids = trace.context.final_llm_chunk_ids
    # Backward-compatible support for hand-authored traces created before
    # final_llm_chunk_ids existed.  The production tracer writes an explicit
    # empty list when no frame reached VLM; a citation in an old trace is the
    # only safe signal that its selected context was consumed.
    legacy_generation_observed = (
        not trace.generation.executed
        and trace.generation.model not in {"", "not_applicable"}
        and bool(trace.generation.answer or trace.generation.citation_ids)
    )
    if not final_llm_ids and (trace.generation.executed or legacy_generation_observed) and trace.generation.citation_ids:
        final_llm_ids = trace.context.selected_chunk_ids
    final_llm = any(matches_context_id(cid) for cid in final_llm_ids)
    exclusion_reason = None
    for exc in trace.context.excluded_chunks:
        if matches(exc.chunk_id, exc.document_id):
            exclusion_reason = exc.exclusion_reason
            break

    events.append(
        EvidenceStageEvent(
            stage="context_selection",
            present=selected if context_applicable else None,
            details={
                "applicable": context_applicable,
                "selection_cutoff_k": trace.context.selection_cutoff_k,
                "token_budget": trace.context.token_budget,
                "selection_method": trace.context.selection_method,
                "final_llm_present": final_llm if context_applicable else None,
                "truncated": trace.context.truncated,
            },
            exclusion_reason=exclusion_reason if context_applicable and not selected else None,
        )
    )

    # 4. Final LLM Generation & Citation
    cited = any(matches_context_id(cit) for cit in trace.generation.citation_ids)
    generation_observed = trace.generation.executed or legacy_generation_observed
    generation_present = (final_llm if context_applicable else None) if generation_observed else None
    events.append(
        EvidenceStageEvent(
            stage="generation",
            present=generation_present,
            details={
                "executed": generation_observed,
                "input_present": final_llm if generation_observed and context_applicable else None,
                "cited": cited,
                "model": trace.generation.model,
                "result_type": trace.generation.result_type,
            },
        )
    )

    # Deducing final disposition & concise explanation
    if not retrieval_hit:
        disposition = "retrieval_miss"
        explanation = f"Evidence ({chunk_id or document_id}) was not retrieved in top-{trace.retrieval.top_k} initial results."
    elif context_applicable and not selected:
        if rerank_hit and rerank_hit.rank_change < 0:
            disposition = "demoted_by_reranker"
            explanation = f"Evidence was retrieved at rank {retrieval_hit.rank}, then observed at rerank rank {rerank_hit.new_rank} and excluded from context."
        else:
            disposition = "excluded_by_context_cutoff"
            explanation = f"Evidence was retrieved at rank {retrieval_hit.rank}, but excluded by context cutoff/budget ({exclusion_reason or 'outside top-K'})."
    elif context_applicable and not final_llm:
        disposition = "not_reached_llm"
        explanation = "Evidence was selected for context but was not observed in the actual generation input."
    elif not generation_observed:
        disposition = "not_applicable"
        explanation = "This query path did not execute a language-model generation stage."
    elif cited:
        disposition = "cited"
        explanation = "Evidence reached the observed generation input and was cited in the generated result."
    elif context_applicable and final_llm:
        disposition = "reached_llm"
        explanation = "Evidence reached the observed generation input but was not explicitly cited."
    else:
        disposition = "reached_result"
        explanation = "Evidence was present in the final ranked result, but no LLM context was applicable."

    return EvidenceLifecycleTrace(
        chunk_id=chunk_id,
        document_id=document_id,
        events=events,
        final_disposition=disposition,
        explanation=explanation,
    )

import json
import re

from search.conversational_context import build_cqr_prompt, build_clarification_prompt

class QueryProcessor:
    """
    Handles CQR (Conversational Query Rewriting), HyDE (Hypothetical Document Embeddings),
    and Query Decomposition for VQA.
    """
    def __init__(self, vlm_client):
        self.vlm = vlm_client

    def rewrite_query_cqr(self, query: str, context_history: list = None) -> str:
        """
        Rewrite conversational queries to include full context. Few-shot
        prompt (PG-ICL, arXiv:2502.15009) built by
        search.conversational_context.build_cqr_prompt - same single VLM
        call as before, just a richer/better-structured prompt. `history`
        turns may now also carry `accepted`/`rejected` feedback descriptions
        (Exquisitor VBS 2024/2025-inspired, prompt-only - see
        record_feedback_in_history), rendered into the prompt by
        format_history.
        """
        if not context_history:
            return query

        prompt = build_cqr_prompt(query, context_history)
        rewritten = self.vlm.generate(None, prompt).strip()
        print(f"CQR Rewrite: '{query}' -> '{rewritten}'")
        return rewritten

    def generate_hyde(self, query: str) -> str:
        """
        Generate a hypothetical document/frame description matching the query.
        """
        prompt = f"""
You are an expert video retrieval assistant. Write a short, factual, and detailed 2-sentence description of what a video keyframe matching this search query would look like. Be concrete about typical objects, colors, actions, and settings. Vietnamese is OK.
Query: "{query}"
Hypothetical description:"""
        
        hyde_answer = self.vlm.generate(None, prompt).strip()
        print(f"HyDE description: '{hyde_answer}'")
        return hyde_answer

    def decompose_query(self, query: str) -> dict:
        """
        Decompose a VQA query into sub-queries and constraints.
        Example query: "người mặc áo đỏ đang đi cạnh chiếc xe màu xanh"
        Decomposes to:
        {
          "sub_queries": ["người mặc áo đỏ", "xe màu xanh"],
          "constraints": ["cùng frame", "đứng cạnh nhau"]
        }
        """
        prompt = f"""
Given a multimodal search query, analyze and split it into:
1. "sub_queries": List of specific visual objects/attributes to detect separately (e.g., ["người mặc áo đỏ", "xe màu xanh"]).
2. "constraints": Spatial or logical relationship constraints (e.g., ["cùng frame", "đứng cạnh nhau"]).
Output ONLY valid JSON matching this format:
{{
  "sub_queries": ["...", "..."],
  "constraints": ["..."]
}}
Query: "{query}"
JSON:"""
        
        raw_output = self.vlm.generate(None, prompt).strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
            
        try:
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            # Fallback decomposition
            return {
                "sub_queries": [query],
                "constraints": []
            }

    def generate_clarification_question(self, query: str, candidate_summaries: list) -> str:
        """
        KIS-C-inspired (VBS_GUIDE.md §4.1: "chat/conversational" task
        variant models a searcher progressively eliciting detail from a
        person who remembers a clip vaguely). When the initial result set
        looks ambiguous (see HybridSearcher.compute_ambiguity_score), this
        generates ONE short clarifying question the operator can put to
        the moderator/participant to narrow down which candidate is
        actually meant - a system-initiated complement to the existing
        passive CQR (which only resolves references already given, rather
        than proactively asking for missing detail).

        Facet-driven prompt (Sekulic et al. zero-shot variant + referring-
        expression-generation disambiguation, see
        search.conversational_context.build_clarification_prompt): the LLM
        first identifies which attribute actually differs across the given
        candidates, then asks about exactly that, instead of a generic
        question - still one VLM call.
        """
        prompt = build_clarification_prompt(query, candidate_summaries)
        question = self.vlm.generate(None, prompt).strip()
        print(f"Clarification question: '{question}'")
        return question

    def decompose_temporal_events(self, query: str) -> list:
        """
        Decompose a TRAKE (Type 3, temporal-alignment) query into an ORDERED
        list of short visual sub-event descriptions, in the exact
        chronological order they occur - e.g. the PDF's "nhảy cao" example
        decomposes to ["chạy đà (bàn chân chạm đất, bước qua vạch xuất
        phát)", "giậm nhảy (chân giậm nhảy rời khỏi mặt đất)", "bay qua xà
        (hông ở vị trí cao nhất)", "tiếp đất (lưng chạm đệm)"]. Order in the
        returned list is load-bearing - Reranker.rerank_type3_temporal's DP
        alignment assumes event i must occur no later than event i+1.
        """
        prompt = f"""
Given a query describing a sequence of chronological events/moments in a video, split it into an ORDERED list of short visual descriptions, one per event, in the exact order they happen.
Output ONLY valid JSON matching this format:
{{"events": ["...", "...", "..."]}}
Query: "{query}"
JSON:"""

        raw_output = self.vlm.generate(None, prompt).strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]

        try:
            parsed = json.loads(raw_output.strip())
            events = parsed.get("events", [])
            return events if events else [query]
        except json.JSONDecodeError:
            return [query]

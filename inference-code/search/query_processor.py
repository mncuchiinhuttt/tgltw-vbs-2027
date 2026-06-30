import json
import re

class QueryProcessor:
    """
    Handles CQR (Conversational Query Rewriting), HyDE (Hypothetical Document Embeddings),
    and Query Decomposition for VQA.
    """
    def __init__(self, vlm_client):
        self.vlm = vlm_client

    def rewrite_query_cqr(self, query: str, context_history: list = None) -> str:
        """
        Rewrite conversational queries to include full context.
        """
        if not context_history:
            return query
            
        history_str = "\n".join([f"User: {turn['query']}\nSystem: {turn.get('answer', '')}" for turn in context_history])
        prompt = f"""
You are a Query Rewriter. Given the conversation history and the latest user query, rewrite the latest query to be fully self-contained and descriptive, resolving any pronouns or implicit references. Keep it concise.
History:
{history_str}
Latest Query: {query}
Rewritten Query:"""
        
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

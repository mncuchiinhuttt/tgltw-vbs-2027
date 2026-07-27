import numpy as np
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, TOP_K_RETRIEVAL, RRF_CONSTANT

class HybridSearcher:
    """
    Performs hybrid search combining Qdrant dense vector search
    with sparse exact text matching on Qdrant payloads, merged via RRF.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )

    def dense_search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
        """
        Search visual index using QwenVL8BEmbedder text encoder.
        """
        query_vector = self.embedder.embed_text(query)
        search_result = self.client.query_points(
            collection_name="visual_index",
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(key="modality", match=MatchValue(value="visual"))
                ]
            )
        ).points
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in search_result
        ]

    def sparse_search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
        """
        Simulated BM25/keyword match using Qdrant's payload full-text search capability.
        Queries the 'text_blob' payload field containing OCR, captions, and labels.
        """
        # Qdrant supports full-text match filtering:
        search_result = self.client.scroll(
            collection_name="visual_index",
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="modality", match=MatchValue(value="visual")),
                    FieldCondition(key="text_blob", match=MatchText(text=query))
                ]
            ),
            limit=top_k,
            with_payload=True
        )[0]
        
        # Qdrant scroll doesn't give a score, so we assign rank based on retrieval order
        return [
            {
                "id": hit.id,
                "score": 1.0 / (idx + 1), # artificial score
                "payload": hit.payload
            } for idx, hit in enumerate(search_result)
        ]

    def merge_rrf(self, dense_hits: list, sparse_hits: list, k: int = RRF_CONSTANT) -> list:
        """
        Reciprocal Rank Fusion (RRF) to merge dense and sparse results.
        Formula: RRF_score(d) = sum_{m in models} 1 / (k + rank_m(d))
        """
        rrf_scores = {}
        payload_map = {}
        
        # Dense ranking
        for rank, hit in enumerate(dense_hits):
            doc_id = hit["id"]
            payload_map[doc_id] = hit["payload"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
            
        # Sparse ranking
        for rank, hit in enumerate(sparse_hits):
            doc_id = hit["id"]
            payload_map[doc_id] = hit["payload"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
            
        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        merged_results = []
        for doc_id, rrf_score in sorted_ids:
            merged_results.append({
                "id": doc_id,
                "rrf_score": rrf_score,
                "payload": payload_map[doc_id]
            })
            
        return merged_results

    def diversify_by_scene(self, candidates: list, top_k: int) -> list:
        """
        Result Diversification: collapses candidates down to the
        highest-RRF-scoring one per (source_file, scene_id) before reranking,
        so top-K isn't flooded by several near-duplicate keyframes from the
        same scene/event instead of covering distinct events (Khoa: Adaptive
        Sampling & Retrieval Accuracy, merged into "Our method" -> Result
        Diversification). Apply this right after merge_rrf, before slicing
        candidates for Stage 3 reranking - for all three query types.
        """
        seen_scenes = set()
        diversified = []
        for hit in sorted(candidates, key=lambda h: h["rrf_score"], reverse=True):
            payload = hit["payload"]
            key = (payload.get("source_file"), payload.get("scene_id"))
            if key in seen_scenes:
                continue
            seen_scenes.add(key)
            diversified.append(hit)
            if len(diversified) >= top_k:
                break
        return diversified

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
        """
        Perform hybrid search query and return fused rankings.
        """
        dense_hits = self.dense_search(query, top_k)
        sparse_hits = self.sparse_search(query, top_k)
        return self.merge_rrf(dense_hits, sparse_hits)

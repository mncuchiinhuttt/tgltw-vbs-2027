import numpy as np
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, SearchParams

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, TOP_K_RETRIEVAL, RRF_CONSTANT,
    SECONDARY_EMBEDDER_ENABLED, QDRANT_EXACT_SEARCH,
)

class HybridSearcher:
    """
    Performs hybrid search combining Qdrant dense vector search
    with sparse exact text matching on Qdrant payloads, merged via RRF.
    """
    def __init__(self, embedder, secondary_embedder=None):
        self.embedder = embedder
        # Fusionista2.0/VERGE-inspired secondary embedding ensemble (see
        # models/siglip_embedder.py) - only meaningful if preprocessing was
        # actually run with SECONDARY_EMBEDDER_ENABLED, populating the
        # "visual_index" collection's named "siglip" vector.
        self.secondary_embedder = secondary_embedder if SECONDARY_EMBEDDER_ENABLED else None
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )

    def dense_search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
        """
        Search visual index using QwenVL8BEmbedder text encoder.

        When SECONDARY_EMBEDDER_ENABLED, "visual_index" was (re)built with
        named vectors ("default" + "siglip" - see preprocessing/indexing/
        indexer.py), so the primary vector must be selected explicitly via
        `using="default"`; a plain (unnamed) single-vector collection
        rejects a `using` param entirely, so it's only passed when enabled.

        AIC's Sơ tuyển round submits a batch of queries within a 4-hour
        window rather than under VBS-style live per-query latency pressure,
        so QDRANT_EXACT_SEARCH defaults to a full brute-force scan instead
        of Qdrant's default approximate HNSW search - see config.py for the
        rationale/citation. Falls back to approximate search (default
        Qdrant behavior) if QDRANT_EXACT_SEARCH is disabled.
        """
        query_vector = self.embedder.embed_text(query)
        search_kwargs = {"using": "default"} if SECONDARY_EMBEDDER_ENABLED else {}
        search_result = self.client.query_points(
            collection_name="visual_index",
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(key="modality", match=MatchValue(value="visual"))
                ]
            ),
            search_params=SearchParams(exact=QDRANT_EXACT_SEARCH),
            **search_kwargs,
        ).points
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in search_result
        ]

    def dense_search_secondary(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
        """
        Second embedding model's (SigLIP) dense search against the
        "visual_index" collection's named "siglip" vector - see
        models/siglip_embedder.py for the ensemble rationale. Returns []
        when no secondary_embedder was provided (disabled), so callers can
        unconditionally include it in a merge_rrf(...) call without an
        extra branch.
        """
        if self.secondary_embedder is None:
            return []
        query_vector = self.secondary_embedder.embed_text(query)
        search_result = self.client.query_points(
            collection_name="visual_index",
            query=query_vector.tolist(),
            using="siglip",
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(key="modality", match=MatchValue(value="visual"))
                ]
            ),
            search_params=SearchParams(exact=QDRANT_EXACT_SEARCH),
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

    def merge_rrf(self, *ranked_lists: list, k: int = RRF_CONSTANT) -> list:
        """
        Reciprocal Rank Fusion (RRF) to merge an arbitrary number of ranked
        hit lists - e.g. dense text-query hits, HyDE hits, and (when
        SECONDARY_EMBEDDER_ENABLED) a second embedding model's dense hits
        via dense_search_secondary (Fusionista2.0/VERGE-inspired ensemble,
        VBS2026 - MMM 2026 LNCS 16415 ch.17/24) - into one fused ranking.
        Purely rank-position based (not raw score), so lists from different
        scoring scales/models combine safely without needing to normalize
        them onto a common range first.
        Formula: RRF_score(d) = sum_{m in models} 1 / (k + rank_m(d))
        Still callable exactly as before with 2 positional lists
        (merge_rrf(dense_hits, sparse_hits)) - k stays keyword-only, no
        existing caller passed it positionally.
        """
        rrf_scores = {}
        payload_map = {}

        for hits in ranked_lists:
            for rank, hit in enumerate(hits):
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

    def get_all_points_for_video(self, video_name: str, limit: int = 10000) -> list:
        """
        Fetches every indexed visual point for a single video, not just
        whatever made it into the initial top-K hybrid-search candidate pool
        - needed for TRAKE's alignment stage (Reranker.rerank_type3_temporal),
        which must consider the video's full frame timeline rather than only
        the handful of frames that happened to score well enough to reach the
        candidate pool. Includes stored vectors (with_vectors=True) so
        callers can compute similarity without re-embedding frame images.
        Single scroll call with a generous limit rather than full pagination
        - a reasonable simplification since one video's keyframe count is
        typically at most a few hundred, well under the default limit.
        """
        points, _ = self.client.scroll(
            collection_name="visual_index",
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="modality", match=MatchValue(value="visual")),
                    FieldCondition(key="source_file", match=MatchValue(value=video_name)),
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        # When SECONDARY_EMBEDDER_ENABLED, "visual_index" uses named vectors
        # and qdrant-client returns p.vector as {"default": [...], "siglip":
        # [...]} instead of a plain list - normalize back to the primary
        # ("default") vector so existing consumers (TRAKE's DP alignment,
        # in_video_refine) keep working unchanged either way.
        def _primary_vector(v):
            return v.get("default") if isinstance(v, dict) else v
        return [{"id": p.id, "payload": p.payload, "vector": _primary_vector(p.vector)} for p in points]

    def in_video_refine(self, query: str, candidates: list, top_videos: int = 5, top_frames_per_video: int = 5) -> list:
        """
        NII-UIT-inspired (VBS2026 winning-lineage system, MMM 2026 LNCS
        16415 ch.26) "In-Video Retrieval": once a handful of candidate
        videos have been identified by the initial hybrid search, re-search
        each video's FULL indexed frame timeline (via get_all_points_for_
        video, not just whatever made the initial top-K candidate pool)
        directly against the query embedding. Surfaces frames that hold the
        actual answer but scored too low on the initial dense+sparse pass
        to make the candidate pool at all - e.g. a frame with the right
        visual content but a generic/unrelated caption. Newly-found frames
        are merged into `candidates` (deduped by point id; an id already in
        `candidates` keeps its existing entry rather than being
        overwritten), not used to replace it.

        Intended for Type 2 (VQA): call this after the initial hybrid
        search + diversify_by_scene pass, before rerank_type2_vqa's
        crop-reranking - it targets Type 2's frame-level localization gap
        specifically, not Type 1's already-broader candidate pool.
        """
        if not candidates:
            return candidates

        video_scores = {}
        for hit in candidates:
            video = hit["payload"].get("source_file")
            if video is None:
                continue
            video_scores[video] = max(video_scores.get(video, 0.0), hit.get("rrf_score", 0.0))
        top_videos_list = sorted(video_scores, key=video_scores.get, reverse=True)[:top_videos]
        if not top_videos_list:
            return candidates

        query_vector = np.asarray(self.embedder.embed_text(query))
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            return candidates

        merged_by_id = {hit["id"]: hit for hit in candidates}
        for video in top_videos_list:
            video_points = self.get_all_points_for_video(video)
            scored_points = []
            for p in video_points:
                if p["id"] in merged_by_id or p.get("vector") is None:
                    continue
                vec = np.asarray(p["vector"])
                vec_norm = float(np.linalg.norm(vec))
                if vec_norm == 0.0:
                    continue
                sim = float(np.dot(query_vector, vec) / (query_norm * vec_norm))
                scored_points.append((sim, p))
            scored_points.sort(key=lambda x: x[0], reverse=True)
            for sim, p in scored_points[:top_frames_per_video]:
                # Scale the raw cosine similarity into the same rrf_score
                # range existing candidates use (rank-1 RRF hit = 1/(k+1))
                # so newly-discovered frames compete fairly in the
                # rrf_score-sorted flow without a second RRF pass.
                merged_by_id[p["id"]] = {
                    "id": p["id"],
                    "rrf_score": sim * (1.0 / (RRF_CONSTANT + 1)),
                    "payload": p["payload"],
                }

        return sorted(merged_by_id.values(), key=lambda h: h.get("rrf_score", 0.0), reverse=True)

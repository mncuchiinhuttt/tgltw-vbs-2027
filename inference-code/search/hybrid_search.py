import numpy as np
import sys
from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, SearchParams

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, TOP_K_RETRIEVAL, RRF_CONSTANT,
    SECONDARY_EMBEDDER_ENABLED, QDRANT_EXACT_SEARCH,
    HEAGLE_LITE_ENABLED, HEAGLE_SHOT_TOP_K, HEAGLE_FRAME_MULTIPLIER,
    DENSE_MAX_PER_SCENE, SPARSE_TOP_K_RETRIEVAL,
    REGION_SEARCH_ENABLED, REGION_SEARCH_TOP_K,
    QUERY_TIME_EXTRACTION_ENABLED, VIDEO_SOURCE_DIR,
    QUERY_TIME_EXTRACTION_FPS, QUERY_TIME_EXTRACTION_MAX_FRAMES,
)
from search.kis_c_scoring import distinct_video_ratio, score_margin_ambiguity, combine_ambiguity_signals
from search.query_time_frames import extract_query_time_frames


def cap_hits_per_scene(hits: list, max_per_scene: int) -> list:
    """
    Keep at most `max_per_scene` hits from any one (video, scene).

    With several frames indexed per shot, a single strongly-matching moment
    can fill the dense pool with near-duplicates of itself and crowd out every
    other scene before diversify_by_scene ever gets to look at them. Capping
    preserves the extra recall the wider index buys without letting one moment
    consume the candidate budget.
    """
    if max_per_scene <= 0:
        return hits
    seen: dict = {}
    capped = []
    for hit in hits:
        payload = hit.get("payload") or {}
        key = (payload.get("source_file"), payload.get("scene_id"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= max_per_scene:
            capped.append(hit)
    return capped

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

    def _dense_search_vector(
        self, vector, top_k: int, exact: Optional[bool] = None, hnsw_ef: Optional[int] = None,
        shot_ids: Optional[list[str]] = None,
    ) -> list:
        """
        Shared primary-vector search core: query_points against the
        "default" named vector (or the collection's sole unnamed vector
        when SECONDARY_EMBEDDER_ENABLED is off), given an already-computed
        query vector. dense_search (text) and dense_search_by_vector
        (pre-computed, e.g. relevance feedback or query-by-example) both
        funnel through this so the actual Qdrant call logic lives in one
        place.

        `exact`, when not None, overrides the QDRANT_EXACT_SEARCH config
        default for just this call - lets an operator escalate precision
        on-demand for one stuck query (U-Cker/PraK-inspired, VBS2026)
        without editing .env/restarting the backend.

        `hnsw_ef` (Qdrant's own documented search-time HNSW candidate-list
        size param, `qdrant_client.models.SearchParams(hnsw_ef=...)`) is a
        graduated middle ground between the binary approximate/exact
        toggle above - raising it trades some latency for higher recall
        without paying exact search's full brute-force cost. Ignored by
        Qdrant when `exact=True` (exact search doesn't traverse the HNSW
        graph at all). None (default) leaves Qdrant's own index-build-time
        default in effect.
        """
        vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        search_kwargs = {"using": "default"} if SECONDARY_EMBEDDER_ENABLED else {}
        use_exact = exact if exact is not None else QDRANT_EXACT_SEARCH
        filter_must = [FieldCondition(key="modality", match=MatchValue(value="visual"))]
        shot_filter = None
        if shot_ids:
            # Use a portable OR-of-MatchValue filter instead of relying on a
            # newer qdrant-client MatchAny model.
            shot_filter = [FieldCondition(key="shot_id", match=MatchValue(value=shot_id)) for shot_id in shot_ids]
        if shot_filter:
            # Qdrant's top-level `should` is optional when `must` already
            # matches. Nest the OR as a required condition so coarse routing
            # genuinely restricts the frame search to selected shots.
            search_filter = Filter(must=filter_must + [Filter(should=shot_filter)])
        else:
            search_filter = Filter(must=filter_must)
        search_result = self.client.query_points(
            collection_name="visual_index",
            query=vector_list,
            limit=top_k,
            query_filter=search_filter,
            search_params=SearchParams(exact=use_exact, hnsw_ef=hnsw_ef),
            **search_kwargs,
        ).points
        hits = [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in search_result
        ]
        return cap_hits_per_scene(hits, DENSE_MAX_PER_SCENE)

    def dense_search(
        self, query: str, top_k: int = TOP_K_RETRIEVAL, exact: Optional[bool] = None, hnsw_ef: Optional[int] = None
    ) -> list:
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
        Qdrant behavior) if QDRANT_EXACT_SEARCH is disabled. Pass `exact`/
        `hnsw_ef` to override the default for just this call.
        """
        vector = self.embedder.embed_text(query)
        if HEAGLE_LITE_ENABLED:
            return self.coarse_to_fine_dense_search(query, vector, top_k, exact=exact, hnsw_ef=hnsw_ef)
        return self._dense_search_vector(vector, top_k, exact=exact, hnsw_ef=hnsw_ef)

    def shot_search(self, query: str, top_k: int = HEAGLE_SHOT_TOP_K) -> list:
        """Retrieve H-EAGLE-lite shot parents for coarse routing."""
        try:
            return self._shot_search_vector(self.embedder.embed_text(query), top_k)
        except Exception as exc:
            print(f"Warning: H-EAGLE-lite shot search unavailable ({exc}); returning no shot hits.")
            return []

    def _shot_search_vector(self, query_vector, top_k: int = HEAGLE_SHOT_TOP_K) -> list:
        search_result = self.client.query_points(
            collection_name="vbs_shot_index",
            query=query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector),
            limit=top_k,
            search_params=SearchParams(exact=QDRANT_EXACT_SEARCH),
        ).points
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in search_result
        ]

    def coarse_to_fine_dense_search(
        self,
        query: str,
        query_vector,
        top_k: int = TOP_K_RETRIEVAL,
        exact: Optional[bool] = None,
        hnsw_ef: Optional[int] = None,
    ) -> list:
        """Search likely shot groups, then search their frame children."""
        del query  # the already-computed vector is the source of truth here
        try:
            shot_result = self._shot_search_vector(query_vector, HEAGLE_SHOT_TOP_K)
        except Exception as exc:
            # A pre-existing deployment may not have run the new shot-index
            # job yet.  H-EAGLE-lite must degrade to the proven frame route,
            # not turn the query endpoint into an outage.
            print(f"Warning: H-EAGLE-lite shot search unavailable ({exc}); using frame search.")
            return self._dense_search_vector(query_vector, top_k, exact=exact, hnsw_ef=hnsw_ef)
        shot_ids = list(dict.fromkeys(
            hit["payload"].get("shot_id") for hit in shot_result if hit.get("payload") and hit["payload"].get("shot_id")
        ))
        if not shot_ids:
            return self._dense_search_vector(query_vector, top_k, exact=exact, hnsw_ef=hnsw_ef)
        return self._dense_search_vector(
            query_vector,
            max(top_k, top_k * HEAGLE_FRAME_MULTIPLIER),
            exact=exact,
            hnsw_ef=hnsw_ef,
            shot_ids=shot_ids,
        )[:top_k]

    def dense_search_by_vector(
        self, vector, top_k: int = TOP_K_RETRIEVAL, exact: Optional[bool] = None, hnsw_ef: Optional[int] = None
    ) -> list:
        """
        Same as dense_search, but takes an already-computed dense vector
        instead of embedding text - used for VBS interactive session
        actions (Phase C) that build their own query vector: relevance
        feedback's Rocchio-adjusted vector, or query-by-example's vector
        retrieved directly from an existing Qdrant point (no re-embedding
        needed).
        """
        if HEAGLE_LITE_ENABLED:
            return self.coarse_to_fine_dense_search("<precomputed-vector>", vector, top_k, exact=exact, hnsw_ef=hnsw_ef)
        return self._dense_search_vector(vector, top_k, exact=exact, hnsw_ef=hnsw_ef)

    def get_point_vector(self, point_id):
        """
        Fetches the stored primary ("default") dense vector for a single
        already-indexed point - used by query-by-example to reuse a
        result's own embedding as the next query without re-running the
        image through the embedder. Returns None if the point doesn't
        exist or has no vector stored.
        """
        points = self.client.retrieve(collection_name="visual_index", ids=[point_id], with_vectors=True)
        if not points:
            return None
        vector = points[0].vector
        return vector.get("default") if isinstance(vector, dict) else vector

    def dense_search_secondary(
        self, query: str, top_k: int = TOP_K_RETRIEVAL, exact: Optional[bool] = None, hnsw_ef: Optional[int] = None
    ) -> list:
        """
        Second embedding model's (SigLIP) dense search against the
        "visual_index" collection's named "siglip" vector - see
        models/siglip_embedder.py for the ensemble rationale. Returns []
        when no secondary_embedder was provided (disabled), so callers can
        unconditionally include it in a merge_rrf(...) call without an
        extra branch. `exact`/`hnsw_ef` override QDRANT_EXACT_SEARCH/the
        HNSW default for this call, same as dense_search.
        """
        if self.secondary_embedder is None:
            return []
        use_exact = exact if exact is not None else QDRANT_EXACT_SEARCH
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
            search_params=SearchParams(exact=use_exact, hnsw_ef=hnsw_ef),
        ).points
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in search_result
        ]

    def dense_search_regions(self, query: str, top_k: int = REGION_SEARCH_TOP_K) -> list:
        """
        Search the region crops preprocessing indexed with modality="region".

        A pooled frame embedding averages a small object away, so a query for
        a license plate or a shop sign scores poorly against the whole frame
        even when the object is plainly there. The crop embeds it on its own.

        Every hit is remapped onto its PARENT frame's point id before being
        returned, which is what keeps regions out of the mechanisms that key
        off frame identity: a crop can promote the frame it came from, but it
        can never appear as a result in its own right, evict its parent from
        the diversified grid, or add a duplicate frame index to TRAKE's
        timeline. Returns [] when region indexing was never run, so callers
        can include it in merge_rrf unconditionally.
        """
        if not REGION_SEARCH_ENABLED:
            return []
        try:
            query_vector = self.embedder.embed_text(query)
            search_result = self.client.query_points(
                collection_name="visual_index",
                query=query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector),
                limit=top_k,
                query_filter=Filter(
                    must=[FieldCondition(key="modality", match=MatchValue(value="region"))]
                ),
                search_params=SearchParams(exact=QDRANT_EXACT_SEARCH),
                **({"using": "default"} if SECONDARY_EMBEDDER_ENABLED else {}),
            ).points
        except Exception as exc:
            print(f"Warning: region search unavailable ({exc}); continuing without it.")
            return []

        # Several crops of one frame collapse to a single parent hit, ranked
        # by their best crop, so a frame with many regions is not rewarded for
        # quantity alone.
        best_by_parent: dict = {}
        for hit in search_result:
            payload = hit.payload or {}
            parent_id = payload.get("parent_point_id")
            if parent_id is None:
                continue
            existing = best_by_parent.get(parent_id)
            if existing is None or hit.score > existing[0]:
                best_by_parent[parent_id] = (hit.score, payload.get("region_concept") or "region")
        if not best_by_parent:
            return []

        # Return the PARENT frame's payload, not the crop's. merge_rrf keys
        # payloads by point id and lets the last writer win, so handing back a
        # region payload under a frame's id would replace that frame's
        # caption, OCR text and detections everywhere downstream.
        try:
            parents = self.client.retrieve(
                collection_name="visual_index", ids=list(best_by_parent), with_payload=True
            )
        except Exception as exc:
            print(f"Warning: could not resolve region parents ({exc}); continuing without them.")
            return []

        hits = []
        for parent in parents:
            score, concept = best_by_parent.get(parent.id, (None, None))
            if score is None:
                continue
            hits.append({
                "id": parent.id,
                "score": score,
                "payload": parent.payload,
                "matched_region": concept,
            })
        return sorted(hits, key=lambda hit: hit["score"], reverse=True)

    def sparse_search(self, query: str, top_k: int = SPARSE_TOP_K_RETRIEVAL) -> list:
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

    def merge_rrf(self, *ranked_lists: list, k: int = RRF_CONSTANT, labels: list = None) -> list:
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

        `labels` (VIREO/SnapMind/NII-UIT-inspired explainability, VBS2026):
        optional list of human-readable names, one per positional
        ranked_lists entry (e.g. ["query", "hyde", "secondary"]). When
        provided, each fused hit gets a `matched_via` field listing which
        named source(s) it appeared in, so an operator can see WHY a result
        matched instead of just a single opaque fused score. Omitted by
        default (None) so the 4 existing callers (CLI main.py, batch_query.py,
        evaluation/run_eval.py, and this class's own search()) keep their
        exact prior behavior unchanged - only webapp/backend/main.py's
        /api/search opts in.
        """
        if labels is not None and len(labels) != len(ranked_lists):
            raise ValueError("labels must have the same length as ranked_lists")

        rrf_scores = {}
        payload_map = {}
        matched_via = {}

        for list_idx, hits in enumerate(ranked_lists):
            for rank, hit in enumerate(hits):
                doc_id = hit["id"]
                payload_map[doc_id] = hit["payload"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))
                if labels is not None:
                    matched_via.setdefault(doc_id, []).append(labels[list_idx])

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged_results = []
        for doc_id, rrf_score in sorted_ids:
            result = {
                "id": doc_id,
                "rrf_score": rrf_score,
                "payload": payload_map[doc_id]
            }
            if labels is not None:
                result["matched_via"] = matched_via[doc_id]
            merged_results.append(result)

        return merged_results

    def temporal_coherence_boost(
        self, candidates: list, window: int = 10, boost_weight: float = 0.3
    ) -> list:
        """
        TAG-inspired (arXiv:2508.07925, "temporal coherence clustering")
        re-scoring: a real event is usually represented by SEVERAL
        temporally-close keyframes, each independently retrieved by
        merge_rrf with its own moderate rrf_score - but per-frame RRF
        fusion treats every candidate as unrelated, so a true event can end
        up "fragmented" across several marginal individual scores instead
        of standing out. This boosts each candidate's rrf_score using the
        combined rrf_score of every OTHER same-video candidate within
        `window` frames of it - several nearby independent hits become a
        stronger combined signal instead of staying fragmented.

        Run right after merge_rrf, BEFORE diversify_by_scene - diversify_by_
        scene then collapses the now-correctly-boosted cluster down to its
        single best representative, so the two steps compose (this one
        fixes ranking within a cluster; diversify_by_scene then dedupes
        across it) rather than duplicating each other's job.
        """
        by_video = {}
        unboosted = []
        for c in candidates:
            video = c["payload"].get("source_file")
            frame_idx = c["payload"].get("frame_idx")
            if video is None or frame_idx is None:
                unboosted.append(c)
                continue
            by_video.setdefault(video, []).append(c)

        for group in by_video.values():
            for c in group:
                frame_idx = c["payload"]["frame_idx"]
                neighbor_score_sum = sum(
                    other.get("rrf_score", 0.0)
                    for other in group
                    if other is not c and abs(other["payload"]["frame_idx"] - frame_idx) <= window
                )
                c["rrf_score"] = c.get("rrf_score", 0.0) + boost_weight * neighbor_score_sum

        boosted = [c for group in by_video.values() for c in group] + unboosted
        return sorted(boosted, key=lambda c: c.get("rrf_score", 0.0), reverse=True)

    def compute_ambiguity_score(self, candidates: list, top_n: int = 10) -> float:
        """
        Cheap, no-VLM-call ambiguity signal (CAR-inspired, arXiv:2511.14769
        "Cluster-based Adaptive Retrieval"; ACL TrustNLP 2025 ambiguity
        detection for QA) for VBS's KIS-C ("chat/conversational") task
        type, which explicitly models a searcher progressively eliciting
        detail from a vague initial query. A confident, well-specified
        query tends to concentrate its top hits on a small number of
        videos/events; a vague query spreads hits across many unrelated
        videos with no clear winner. Caller (webapp/backend/main.py)
        decides the threshold at which to act on this (e.g. trigger a
        clarification question).

        Blends two signals (search.kis_c_scoring, pure/testable), but at the
        current kis_c_scoring.MARGIN_WEIGHT of 1.0 only the second one has
        any weight:
        - score_margin_ambiguity (weight 1.0): the top-1 vs top-2 score
          margin, which simplifies exactly to top2/top1. Meaningful here
          because this method runs AFTER temporal_coherence_boost, which
          spreads out an otherwise near-flat RRF score distribution. Margin
          was chosen over entropy for simplicity, and entropy was later
          measured to be strictly worse anyway.
        - distinct_video_ratio (weight 0.0): ratio of DISTINCT videos among
          the top `top_n` candidates. Retained for tunability but currently
          inert, because this method also runs AFTER diversify_by_scene -
          which collapses the pool to one candidate per (source_file,
          scene_id), so the ratio sits at a constant ~1.0 and carried a
          fixed offset instead of information. See kis_c_scoring.
          MARGIN_WEIGHT for the measurements.
        Still returns a single float in [0.0, 1.0] on the scale the
        AMBIGUITY_THRESHOLD env knob is compared against - no caller change.
        A pool of fewer than 2 candidates scores 0.0 (the margin is
        undefined) and so never triggers a clarification.
        """
        distinct_ratio = distinct_video_ratio(candidates, top_n)
        margin_ambiguity = score_margin_ambiguity(candidates, top_n)
        combined = combine_ambiguity_signals(distinct_ratio, margin_ambiguity)
        print(f"Ambiguity signals: distinct_ratio={distinct_ratio:.2f}, margin_ambiguity={margin_ambiguity:.2f}, combined={combined:.2f}")
        return combined

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

    def search(
        self, query: str, top_k: int = TOP_K_RETRIEVAL, exact: Optional[bool] = None, hnsw_ef: Optional[int] = None
    ) -> list:
        """
        Perform hybrid search query and return fused rankings. `exact`/
        `hnsw_ef` override QDRANT_EXACT_SEARCH/the HNSW default for the
        dense branch only - sparse_search is a payload text filter, not a
        vector search, so it has no exact/HNSW setting to override.
        """
        dense_hits = self.dense_search(query, top_k, exact=exact, hnsw_ef=hnsw_ef)
        sparse_hits = self.sparse_search(query)
        region_hits = self.dense_search_regions(query)
        return self.merge_rrf(dense_hits, sparse_hits, region_hits)

    def get_all_points_for_video(self, video_name: str, limit: int = 10000) -> list:
        """
        Fetches every indexed visual point for a single video, not just
        whatever made it into the initial top-K hybrid-search candidate pool
        - needed for TRAKE's alignment stage (Reranker.rerank_type3_temporal),
        which must consider the video's full frame timeline rather than only
        the handful of frames that happened to score well enough to reach the
        candidate pool. Includes stored vectors (with_vectors=True) so
        callers can compute similarity without re-embedding frame images.
        Paginated rather than a single generous-limit call: this used to
        assume a video held "at most a few hundred" keyframes, which stopped
        being true once several frames are indexed per shot instead of one. A
        silent truncation here would hide the tail of a long video from
        TRAKE's alignment and from in-video refinement, which is exactly the
        recall the wider index was meant to buy.
        """
        scroll_filter = Filter(
            must=[
                FieldCondition(key="modality", match=MatchValue(value="visual")),
                FieldCondition(key="source_file", match=MatchValue(value=video_name)),
            ]
        )
        points = []
        offset = None
        while len(points) < limit:
            batch, offset = self.client.scroll(
                collection_name="visual_index",
                scroll_filter=scroll_filter,
                limit=min(1000, limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points.extend(batch)
            if offset is None or not batch:
                break
        # When SECONDARY_EMBEDDER_ENABLED, "visual_index" uses named vectors
        # and qdrant-client returns p.vector as {"default": [...], "siglip":
        # [...]} instead of a plain list - normalize back to the primary
        # ("default") vector so existing consumers (TRAKE's DP alignment,
        # in_video_refine) keep working unchanged either way.
        def _primary_vector(v):
            return v.get("default") if isinstance(v, dict) else v
        return [{"id": p.id, "payload": p.payload, "vector": _primary_vector(p.vector)} for p in points]

    def rocchio_adjust(
        self,
        base_vector,
        positive_vectors: list,
        negative_vectors: list,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.15,
    ):
        """
        Classic Rocchio relevance-feedback formula: shifts a query vector
        toward the centroid of vectors marked positive and away from the
        centroid of vectors marked negative, then re-normalizes. Used by
        /api/feedback (webapp/backend/main.py, Phase C) as a lightweight
        relevance-feedback mechanism for VBS's live interactive search -
        pure vector arithmetic, no extra model/VLM call, unlike a full
        Bayesian relevance model (CVHunter/PraK) or LLM-based query
        rewriting.
        new_vector = alpha*base + beta*mean(positive) - gamma*mean(negative),
        re-normalized to unit length.
        """
        base = np.asarray(base_vector, dtype=float)
        result = alpha * base
        if positive_vectors:
            result = result + beta * np.mean(np.asarray(positive_vectors, dtype=float), axis=0)
        if negative_vectors:
            result = result - gamma * np.mean(np.asarray(negative_vectors, dtype=float), axis=0)
        norm = float(np.linalg.norm(result))
        return result / norm if norm > 0 else result

    def temporal_chain_match(self, hit_lists: list, window_frames: int = 150) -> list:
        """
        VBS-style temporal query, generalized to a chain of N>=2 sequential
        text descriptions ("a bicycle passes, then a red car, then a dog
        runs by") searched independently - distinct from AIC's TRAKE which
        takes ONE sentence and auto-decomposes it via DP alignment (see
        Reranker.rerank_type3_temporal). Exquisitor-inspired sequence-chain
        matching: for each video present in every hit list, finds the
        highest-combined-RRF-score chain of frames frame_0 < frame_1 < ... <
        frame_{N-1} where each consecutive pair is within window_frames.

        Implemented as a step-wise DP (same style as
        Reranker._align_events_dp): dp[i][frame] = best cumulative score of
        a valid chain ending with step i assigned to `frame`, built from
        dp[i-1][prev_frame] for every prev_frame within the window
        constraint. Backpointers reconstruct the actual frame sequence.
        With exactly 2 hit lists this reduces to the same best-pair-per-video
        result the old temporal_window_match produced.
        """
        def _group_by_video(hits):
            grouped = {}
            for hit in hits:
                video = hit["payload"].get("source_file")
                frame_idx = hit["payload"].get("frame_idx")
                if video is None or frame_idx is None:
                    continue
                grouped.setdefault(video, []).append(hit)
            return grouped

        by_video_per_step = [_group_by_video(hits) for hits in hit_lists]
        if not by_video_per_step:
            return []
        common_videos = set(by_video_per_step[0])
        for by_video in by_video_per_step[1:]:
            common_videos &= set(by_video)

        matches = []
        for video in common_videos:
            step_hits = [by_video[video] for by_video in by_video_per_step]

            # dp[i] maps frame_idx -> (cumulative_score, hit, backpointer_frame)
            dp = [
                {h["payload"]["frame_idx"]: (h.get("rrf_score", 0.0), h, None) for h in step_hits[0]}
            ]
            for i in range(1, len(step_hits)):
                prev_dp = dp[i - 1]
                step_dp = {}
                for h in step_hits[i]:
                    frame = h["payload"]["frame_idx"]
                    best_prev_frame, best_total = None, -1.0
                    for prev_frame, (prev_score, _, _) in prev_dp.items():
                        if prev_frame < frame and (frame - prev_frame) <= window_frames:
                            total = prev_score + h.get("rrf_score", 0.0)
                            if total > best_total:
                                best_total, best_prev_frame = total, prev_frame
                    if best_prev_frame is not None:
                        # Keep the best-scoring hit reaching this frame if
                        # multiple prior chains land on the same frame.
                        existing = step_dp.get(frame)
                        if existing is None or best_total > existing[0]:
                            step_dp[frame] = (best_total, h, best_prev_frame)
                dp.append(step_dp)

            if not dp[-1]:
                continue  # no valid chain reaches the last step for this video

            last_frame, (best_score, _, _) = max(dp[-1].items(), key=lambda kv: kv[1][0])

            # Reconstruct the chain by walking backpointers.
            chain_hits = []
            frame = last_frame
            for i in range(len(dp) - 1, -1, -1):
                _, hit, prev_frame = dp[i][frame]
                chain_hits.append(hit)
                frame = prev_frame
            chain_hits.reverse()

            matches.append({
                "video_name": video,
                "score": best_score,
                "frames": [h["payload"]["frame_idx"] for h in chain_hits],
                "payloads": [h["payload"] for h in chain_hits],
            })

        return sorted(matches, key=lambda m: m["score"], reverse=True)

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

            # Everything above reorders frames that were already indexed, so
            # it cannot recover a moment offline selection never kept. This
            # decodes the video and scores frames that have no index point at
            # all, which is the only step here that raises the recall ceiling
            # rather than redistributing what is under it.
            for extracted in self._extract_query_time_frames(video, query_vector, video_points, top_frames_per_video):
                merged_by_id[extracted["id"]] = extracted

        return sorted(merged_by_id.values(), key=lambda h: h.get("rrf_score", 0.0), reverse=True)

    def _extract_query_time_frames(self, video: str, query_vector, video_points: list, top_frames: int) -> list:
        """Decode and score frames of `video` that are not in the index."""
        if not QUERY_TIME_EXTRACTION_ENABLED or not VIDEO_SOURCE_DIR:
            return []
        known = {
            point["payload"].get("frame_idx")
            for point in video_points
            if point.get("payload") and point["payload"].get("frame_idx") is not None
        }
        try:
            extracted = extract_query_time_frames(
                video_name=video,
                video_source_dir=VIDEO_SOURCE_DIR,
                embedder=self.embedder,
                query_vector=query_vector,
                known_frame_indices=known,
                sampling_fps=QUERY_TIME_EXTRACTION_FPS,
                max_frames=QUERY_TIME_EXTRACTION_MAX_FRAMES,
                top_frames=top_frames,
            )
        except Exception as exc:
            print(f"Warning: query-time frame extraction failed for {video} ({exc}).")
            return []

        return [
            {
                # Not a Qdrant point - it was decoded just now - so the id is
                # synthetic and marked, and must never be fed back to Qdrant.
                "id": f"query-time:{video}:{frame['payload']['frame_idx']}",
                "rrf_score": frame["similarity"] * (1.0 / (RRF_CONSTANT + 1)),
                "payload": frame["payload"],
            }
            for frame in extracted
        ]

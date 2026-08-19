import json
import re
import numpy as np
from PIL import Image
from typing import Callable, List, Dict, Any, Optional
from config import (
    TRAKE_MAX_VIDEOS_TO_ALIGN,
    VERIFICATION_RERANK_ENABLED, VERIFICATION_NUM_QUESTIONS, VERIFICATION_WEIGHT_TYPE1,
    TYPE2_RRF_WEIGHT, TYPE2_VQA_WEIGHT, TYPE2_VERIFICATION_WEIGHT,
)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def _align_events_dp(similarity_matrix: List[List[float]]) -> List[int]:
    """
    DANTE-inspired (arXiv:2512.13169) subsequence-alignment DP: given an
    N (sub-events) x M (candidate frames, chronologically ordered) similarity
    matrix, finds the assignment of one frame per event - in strictly
    increasing frame-index order, so the N answers come out chronologically
    ordered as TRAKE's answer format requires - maximizing total assigned
    similarity. Classic O(N*M) weighted-subsequence DP (same structure as
    weighted LIS / global sequence alignment). Optimizes globally over the
    whole matrix rather than depending on frames' original retrieval-rank
    order, so it tolerates "temporally-incoherent" retrieval noise the way
    the previous single-holistic-VLM-score approach couldn't.
    Returns a list of N frame-column indices (one per event); an event gets
    -1 if no valid assignment exists (e.g. fewer frames than events).
    """
    n = len(similarity_matrix)
    if n == 0:
        return []
    m = len(similarity_matrix[0]) if similarity_matrix[0] else 0
    if m < n:
        return [-1] * n

    NEG_INF = float("-inf")
    # dp[i][j]: best total similarity aligning events[0..i] using only
    # frames[0..j] (event i assigned to some frame <= j).
    dp = [[NEG_INF] * m for _ in range(n)]
    take_here = [[False] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            best_skip = dp[i][j - 1] if j > 0 else NEG_INF
            if i == 0:
                prior = 0.0
            else:
                prior = dp[i - 1][j - 1] if j > 0 else NEG_INF
            best_take = prior + similarity_matrix[i][j] if prior != NEG_INF else NEG_INF

            if best_take >= best_skip:
                dp[i][j] = best_take
                take_here[i][j] = True
            else:
                dp[i][j] = best_skip
                take_here[i][j] = False

    assigned = [-1] * n
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        if take_here[i][j]:
            assigned[i] = j
            i -= 1
            j -= 1
        else:
            j -= 1
    return assigned

def rerank_with_tail(
    rerank_fn: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    candidates: List[Dict[str, Any]],
    rerank_top_k: int,
    submission_top_k: int,
) -> List[Dict[str, Any]]:
    """
    Runs the (expensive, VLM-based) rerank_fn on just the head of candidates
    (rerank_top_k) and appends the rest (up to submission_top_k) in their
    original retrieval-rank order, instead of VLM-reranking the whole pool.
    The AIC competition's scoring rewards submitting up to 100 ranked answers
    per query (R@1/5/20/50/100, see SUBMISSION_TOP_K in config.py) - this
    lets R@1/5/20 benefit from full VLM reranking quality while R@50/R@100
    still get filled out with more candidates, without paying for a VLM call
    per candidate just to rank the tail.
    """
    head = candidates[:rerank_top_k]
    tail = candidates[rerank_top_k:submission_top_k]
    reranked_head = rerank_fn(head) if head else []
    return reranked_head + tail

def _parse_vlm_score(text: str) -> Optional[float]:
    """
    Extract the first float from a VLM's score response, tolerating extra text
    around it (e.g. "Score: 0.8" or "0.8 - strong match"). Returns None if no
    number can be found, so callers can distinguish a genuine 0.0 score from a
    parse failure instead of silently collapsing both to 0.0.
    """
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group())

class Reranker:
    """
    Implements stage reranking, VQA crop-reranking, and temporal sequence alignment.
    """
    def __init__(self, vlm_client, detector_client=None):
        self.vlm = vlm_client
        self.detector = detector_client

    def generate_verification_questions(self, query: str, n: int = VERIFICATION_NUM_QUESTIONS) -> List[str]:
        """
        Fusionista2.0-inspired (VBS2026, MMM 2026 LNCS 16415 ch.17 -
        "Reranking with Interactive Confirmation") verification pass: ask the
        VLM to break the query down into n short yes/no checks, each on ONE
        specific object/attribute/action it mentions. Called once per query
        (not per-candidate) - the same questions are then checked against
        every candidate by verify_candidate below, instead of trusting one
        holistic similarity score that can be fooled by a partial match.
        """
        prompt = f"""Given a video search query, generate {n} short yes/no verification questions that each check ONE specific object/attribute/action mentioned in the query.
Output ONLY valid JSON matching this format:
{{"questions": ["...", "..."]}}
Query: "{query}"
JSON:"""
        raw_output = self.vlm.generate(None, prompt).strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        try:
            parsed = json.loads(raw_output.strip())
            questions = parsed.get("questions", [])
            return questions[:n] if questions else [query]
        except json.JSONDecodeError:
            return [query]

    def verify_candidate(
        self,
        image: Optional[Image.Image],
        context_text: str,
        questions: List[str],
    ) -> float:
        """
        Checks each verification question against a candidate - against its
        actual image when one is available (Type 2's cropped frame), or
        against its text metadata otherwise (Type 1 has no guaranteed local
        image path, see rerank_type1) - and returns the fraction answered
        YES. This "match ratio" blends into the existing rerank score
        (rerank_type1/rerank_type2_vqa) rather than replacing it, so a
        candidate that fails some but not all checks is dampened, not
        zeroed out.
        """
        if not questions:
            return 1.0
        matches = 0
        for q in questions:
            if image is not None:
                prompt = f"Looking at this image, answer this yes/no question: {q}\nAnswer only YES or NO."
                response = self.vlm.generate(image, prompt).strip().upper()
            else:
                prompt = f"Frame info: {context_text}\nBased on this info, answer this yes/no question: {q}\nAnswer only YES or NO."
                response = self.vlm.generate(None, prompt).strip().upper()
            if response.startswith("YES"):
                matches += 1
        return matches / len(questions)

    def rerank_type1(
        self, query: str, candidate_frames: List[Dict[str, Any]], verify: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using VLM comparing query with frame data.

        `verify`, when not None, overrides VERIFICATION_RERANK_ENABLED for
        this call - lets an operator escalate to verification reranking
        on-demand for a stuck query without editing .env/restarting.
        """
        print(f"Reranking {len(candidate_frames)} candidates for Type 1 query...")
        use_verify = verify if verify is not None else VERIFICATION_RERANK_ENABLED
        questions = self.generate_verification_questions(query) if use_verify else []
        scored = []
        for hit in candidate_frames:
            payload = hit["payload"]

            # Since the raw image path might be needed, we assume image path can be loaded
            # or generated. If we don't have local paths, we fallback to payload data text comparison
            # or load image if we mock/simulate it. Let's write the code assuming image is loaded if available,
            # or run on metadata.
            frame_description = f"Caption: {payload.get('caption', '')}. Narrative: {payload.get('scene_narrative', '')}. OCR: {payload.get('ocr_text', '')}."

            # The negation line matters here specifically: this is the only
            # stage that scores query against candidate with a language model
            # rather than an embedding, and NevIR (Weller et al., EACL
            # 2024, arXiv:2305.07614) finds cross-encoders are the ONLY architecture to beat random on
            # negated pairs - bi-encoder and sparse retrieval, i.e. everything
            # upstream of this call, score below random. If a KIS-C
            # clarification answer's negation survives CQR, this is the last
            # place it can still be honoured.
            prompt = f"""
Query: "{query}"
Frame info: {frame_description}
Compare the query with the frame metadata and rate how well this frame matches the query from 0.0 (no match) to 1.0 (perfect match). If the query says something is NOT present or NOT a given attribute, a frame that does show it must score LOW, not high. Output only the score as a float.
Score:"""

            # Text-only comparison as base/fallback, or vision if image is provided
            score_str = self.vlm.generate(None, prompt).strip()
            score = _parse_vlm_score(score_str)
            if score is None:
                print(f"Warning: could not parse rerank score from VLM response: {score_str!r}. Defaulting to 0.0.")
                score = 0.0
                hit["rerank_score_valid"] = False
            else:
                hit["rerank_score_valid"] = True

            if questions:
                verification_ratio = self.verify_candidate(None, frame_description, questions)
                hit["verification_ratio"] = verification_ratio
                score = (1 - VERIFICATION_WEIGHT_TYPE1) * score + VERIFICATION_WEIGHT_TYPE1 * verification_ratio

            hit["rerank_score"] = score
            scored.append(hit)

        return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)

    def crop_bounding_box(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """
        Crop bounding box [x1, y1, x2, y2] from a PIL image.
        """
        width, height = image.size
        # Bboxes might be normalized or pixel coordinates.
        # Assume pixel coordinates first, but clamp to image bounds.
        x1 = max(0, min(width, int(bbox[0])))
        y1 = max(0, min(height, int(bbox[1])))
        x2 = max(0, min(width, int(bbox[2])))
        y2 = max(0, min(height, int(bbox[3])))
        
        if x2 <= x1 or y2 <= y1:
            return image # return original if bbox invalid
        return image.crop((x1, y1, x2, y2))

    def rerank_type2_vqa(
        self,
        query: str,
        sub_queries: List[str],
        candidate_frames: List[Dict[str, Any]],
        dataset_dir: str,
        verify: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Type 2 Visual Question Answering crop-reranking logic:
        1. Run object detection for sub-queries on each frame.
        2. Crop image around matching bounding boxes if found.
        3. Pass crop (or fallback full frame) to VLM to answer query.
        4. Calculate weighted score: 0.4 * rrf_score + 0.6 * vqa_score.

        `verify`, when not None, overrides VERIFICATION_RERANK_ENABLED for
        this call - see rerank_type1's docstring for the rationale.
        """
        print(f"Executing Type 2 VQA reranking for query: '{query}'...")
        use_verify = verify if verify is not None else VERIFICATION_RERANK_ENABLED
        questions = self.generate_verification_questions(query) if use_verify else []
        scored = []

        for hit in candidate_frames:
            payload = hit["payload"]
            video_name = payload["source_file"]
            timestamp = payload["timestamp"]
            
            # Form path to keyframe image (assumed saved during preprocessing or simulated)
            # In a real environment, keyframe image is loaded from disk.
            # We mock the image loading here, or construct the local file path if present.
            frame_img = None
            frame_path = os.path.join(dataset_dir, video_name) # simplified path
            if os.path.exists(frame_path) and frame_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    frame_img = Image.open(frame_path).convert("RGB")
                except Exception:
                    pass
            
            # If no actual image file is found, create a dummy placeholder PIL image for mock runs
            if frame_img is None:
                frame_img = Image.new("RGB", (640, 480), color=(128, 128, 128))
            
            # Bounding box cropping logic based on sub-queries
            crop_img = frame_img
            if self.detector is not None and sub_queries:
                # Search for target objects matching our sub-queries
                detections = self.detector.detect(frame_img, sub_queries)
                if detections:
                    # Select detection with highest confidence
                    best_det = max(detections, key=lambda x: x["conf"])
                    print(f"Found object '{best_det['label']}' with conf {best_det['conf']}. Cropping bbox {best_det['bbox']}...")
                    crop_img = self.crop_bounding_box(frame_img, best_det["bbox"])
            
            # VQA Evaluation Prompt on the cropped image
            VQA_PROMPT = f"""
            Question: {query}
            Answer YES or NO, then give confidence 0-1.
            Format: {{"answer": "YES/NO", "confidence": 0.9, "reason": "..."}}
            """
            
            vqa_score = 0.0
            try:
                raw_res = self.vlm.generate(crop_img, VQA_PROMPT).strip()
                if raw_res.startswith("```json"):
                    raw_res = raw_res[7:]
                if raw_res.endswith("```"):
                    raw_res = raw_res[:-3]
                    
                result = json.loads(raw_res.strip())
                conf = float(result.get("confidence", 0.5))
                
                if result.get("answer") == "YES":
                    vqa_score = conf
                else:
                    vqa_score = 1.0 - conf
            except Exception as e:
                print(f"VQA scoring failed for frame: {e}")
                vqa_score = 0.5 # neutral score on failure
                
            # Weighted fusion: original_rank (rrf_score), vqa_score, and
            # (if enabled) the verification match ratio against crop_img -
            # falls back to the original rrf/vqa-only split when disabled.
            rrf_score = hit.get("rrf_score", 0.0)
            hit["vqa_score"] = vqa_score
            if questions:
                verification_ratio = self.verify_candidate(crop_img, "", questions)
                hit["verification_ratio"] = verification_ratio
                hit["final_score"] = (
                    TYPE2_RRF_WEIGHT * rrf_score
                    + TYPE2_VQA_WEIGHT * vqa_score
                    + TYPE2_VERIFICATION_WEIGHT * verification_ratio
                )
            else:
                hit["final_score"] = 0.4 * rrf_score + 0.6 * vqa_score
            scored.append(hit)
            
        return sorted(scored, key=lambda x: x["final_score"], reverse=True)

    def rerank_type3_temporal(
        self,
        query: str,
        candidate_frames: List[Dict[str, Any]],
        query_proc,
        searcher,
        max_videos: int = TRAKE_MAX_VIDEOS_TO_ALIGN,
    ) -> List[Dict[str, Any]]:
        """
        DANTE-inspired (arXiv:2512.13169) two-stage TRAKE alignment,
        replacing the previous single holistic VLM score for the whole
        sequence with an actual N-events <-> M-frames alignment:

        Stage 1 (Retrieval): rank candidate VIDEOS by the best rrf_score
        among each video's frame-hits already in candidate_frames (per the
        competition's own "find the one video" framing, generalized here to
        the top `max_videos` rather than exactly one - the AIC scoring rule
        separately rewards submitting up to 100 ranked answers per query, so
        multiple candidate videos each get their own aligned sequence,
        ranked by alignment quality, instead of only ever offering one guess).
        Stage 2 (Alignment): for each of those videos, decompose the query
        into N ordered sub-events (query_proc.decompose_temporal_events) and
        fetch EVERY indexed point of that video (searcher.get_all_points_for_
        video - not just whatever made the initial candidate pool), then run
        a dynamic-programming subsequence alignment (_align_events_dp) over
        the N x M similarity matrix to pick the best chronologically-ordered
        frame per sub-event.
        """
        print("Executing Type 3 Temporal reasoning (DP alignment)...")

        if not candidate_frames:
            return []

        # Stage 1: rank candidate videos by their best frame-hit's rrf_score
        video_scores: Dict[str, float] = {}
        for hit in candidate_frames:
            video = hit["payload"]["source_file"]
            video_scores[video] = max(video_scores.get(video, 0.0), hit.get("rrf_score", 0.0))
        ranked_videos = sorted(video_scores, key=video_scores.get, reverse=True)[:max_videos]

        events = query_proc.decompose_temporal_events(query)
        event_vectors = [np.asarray(searcher.embedder.embed_text(ev)) for ev in events]

        sequences = []
        for video in ranked_videos:
            all_points = searcher.get_all_points_for_video(video)
            all_points = [
                p for p in all_points
                if p["payload"].get("frame_idx") is not None or p["payload"].get("timestamp") is not None
            ]
            all_points.sort(
                key=lambda p: p["payload"]["frame_idx"] if p["payload"].get("frame_idx") is not None
                else p["payload"]["timestamp"]
            )

            if len(all_points) < len(events):
                # Not enough frames in this video to host every sub-event -
                # skip rather than force a degenerate partial alignment.
                continue

            similarity_matrix = [
                [_cosine_sim(ev_vec, np.asarray(p["vector"])) for p in all_points]
                for ev_vec in event_vectors
            ]
            assigned_indices = _align_events_dp(similarity_matrix)
            if any(idx < 0 for idx in assigned_indices):
                continue

            aligned_points = [all_points[j] for j in assigned_indices]
            total_sim = sum(similarity_matrix[i][j] for i, j in enumerate(assigned_indices))

            sequences.append({
                "video_name": video,
                # Native video frame index (see preprocessing/main.py's
                # "frame_idx" payload field), NOT a Qdrant point UUID - that
                # carries no temporal/frame-position meaning at all and would
                # produce a meaningless <frame_id> in the submission format.
                "frame_ids": [p["payload"].get("frame_idx") for p in aligned_points],
                "timestamps": [p["payload"].get("timestamp") for p in aligned_points],
                "score": total_sim / len(events),
                "events": events,
            })

        return sorted(sequences, key=lambda s: s["score"], reverse=True)
import os

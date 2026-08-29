import json
import math
import os
import re
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
    score = float(match.group())
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _strip_json_fence(text: str) -> str:
    """Remove a Markdown JSON fence without accepting extra prose."""
    value = (text or "").strip()
    if value.startswith("```"):
        first_newline = value.find("\n")
        value = value[first_newline + 1:] if first_newline >= 0 else value[3:]
    if value.endswith("```"):
        value = value[:-3]
    return value.strip()


def parse_grounded_vqa_response(raw_response: str) -> Dict[str, Any]:
    """Parse the live VQA contract and fail closed on every invalid response."""
    invalid = {
        "valid": False,
        "found": False,
        "answer": "UNKNOWN",
        "confidence": 0.0,
        "reason": "malformed_response",
    }
    try:
        parsed = json.loads(_strip_json_fence(raw_response))
    except (TypeError, json.JSONDecodeError):
        return invalid
    if not isinstance(parsed, dict):
        return invalid
    if parsed.get("found") is not True:
        return {**invalid, "reason": "answer_not_found"}

    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return {**invalid, "reason": "missing_answer"}
    if answer.strip().upper() in {"UNKNOWN", "N/A", "NA", "NONE", "NULL"}:
        return {**invalid, "reason": "unknown_answer"}

    raw_confidence = parsed.get("confidence")
    if isinstance(raw_confidence, bool):
        return {**invalid, "reason": "invalid_confidence"}
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        return {**invalid, "reason": "invalid_confidence"}
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return {**invalid, "reason": "confidence_out_of_range"}

    return {
        "valid": True,
        "found": True,
        "answer": answer.strip(),
        "confidence": confidence,
        "reason": str(parsed.get("reason") or ""),
    }


def _candidate_paths(dataset_dir: str, payload: Dict[str, Any]) -> List[str]:
    """Return only media paths contained by the configured dataset directory."""
    if not isinstance(payload, dict):
        return []
    try:
        dataset_root = os.path.realpath(dataset_dir)
    except (TypeError, ValueError, OSError):
        return []
    paths: List[str] = []
    for field in ("keyframe_path", "frame_path", "source_file"):
        raw_path = payload.get(field)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = os.path.realpath(raw_path if os.path.isabs(raw_path) else os.path.join(dataset_root, raw_path))
        try:
            if os.path.commonpath((dataset_root, path)) != dataset_root:
                continue
        except ValueError:
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _coerce_frame_idx(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _coerce_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def resolve_candidate_evidence(dataset_dir: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve the exact media and canonical identity used for VQA."""
    for frame_path in _candidate_paths(dataset_dir, payload):
        if not os.path.isfile(frame_path):
            continue
        try:
            with Image.open(frame_path) as image:
                return {
                    "image": image.convert("RGB").copy(),
                    "path": frame_path,
                    "frame_idx": _coerce_frame_idx(payload.get("frame_idx")),
                    "timestamp": _coerce_timestamp(
                        payload.get("timestamp")
                        if payload.get("timestamp") is not None
                        else payload.get("pts_time")
                    ),
                }
        except (OSError, Image.UnidentifiedImageError):
            pass

        capture = cv2.VideoCapture(frame_path)
        if not capture.isOpened():
            capture.release()
            continue
        try:
            # frame_idx is the canonical indexed identity. Timestamp is only
            # a fallback for legacy payloads that do not have it.
            frame_idx = _coerce_frame_idx(payload.get("frame_idx"))
            timestamp = payload.get("timestamp")
            if timestamp is None:
                timestamp = payload.get("pts_time")
            timestamp = _coerce_timestamp(timestamp)
            if frame_idx is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            elif timestamp is not None:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            else:
                continue
            success, frame = capture.read()
            if success and frame is not None:
                actual_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not math.isfinite(actual_timestamp) or actual_timestamp < 0:
                    actual_timestamp = timestamp
                return {
                    "image": Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    "path": frame_path,
                    "frame_idx": frame_idx,
                    "timestamp": actual_timestamp,
                }
        finally:
            capture.release()
    return None


def load_candidate_frame(dataset_dir: str, payload: Dict[str, Any]) -> Optional[Image.Image]:
    """Load a real keyframe or decode its source video at the payload time."""
    evidence = resolve_candidate_evidence(dataset_dir, payload)
    return evidence["image"] if evidence is not None else None

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
        try:
            parsed = json.loads(_strip_json_fence(raw_output))
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(parsed, dict) or not isinstance(parsed.get("questions"), list):
            return []
        questions = [q.strip() for q in parsed["questions"] if isinstance(q, str) and q.strip()]
        return questions if len(questions) == n else []

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

        def _score_single_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
            hit_copy = dict(hit)
            payload = hit_copy["payload"]

            frame_description = f"Caption: {payload.get('caption', '')}. Narrative: {payload.get('scene_narrative', '')}. OCR: {payload.get('ocr_text', '')}."

            prompt = f"""
Query: "{query}"
Frame info: {frame_description}
Compare the query with the frame metadata and rate how well this frame matches the query from 0.0 (no match) to 1.0 (perfect match). Output only the score as a float.
Score:"""

            score_str = self.vlm.generate(None, prompt).strip()
            score = _parse_vlm_score(score_str)
            if score is None:
                print(f"Warning: could not parse rerank score from VLM response: {score_str!r}. Defaulting to 0.0.")
                score = 0.0
                hit_copy["rerank_score_valid"] = False
            else:
                hit_copy["rerank_score_valid"] = True

            if questions:
                verification_ratio = self.verify_candidate(None, frame_description, questions)
                hit_copy["verification_ratio"] = verification_ratio
                score = (1 - VERIFICATION_WEIGHT_TYPE1) * score + VERIFICATION_WEIGHT_TYPE1 * verification_ratio

            rrf_score = float(hit_copy.get("rrf_score", 0.0))
            hit_copy["rerank_score"] = score
            hit_copy["final_score"] = 0.4 * rrf_score + 0.6 * score
            return hit_copy

        if not candidate_frames:
            return []

        max_workers = min(len(candidate_frames), 8)
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                scored = list(executor.map(_score_single_hit, candidate_frames))
        else:
            scored = [_score_single_hit(h) for h in candidate_frames]

        return sorted(scored, key=lambda x: (x.get("final_score", 0.0), x.get("rerank_score", 0.0), x.get("rrf_score", 0.0)), reverse=True)

    def crop_bounding_box(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """
        Crop bounding box [x1, y1, x2, y2] from a PIL image.
        """
        width, height = image.size
        try:
            coords = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return image
        if len(coords) != 4 or not all(math.isfinite(value) for value in coords):
            return image
        if all(0.0 <= value <= 1.0 for value in coords):
            coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]
        x1 = max(0, min(width, int(round(coords[0]))))
        y1 = max(0, min(height, int(round(coords[1]))))
        x2 = max(0, min(width, int(round(coords[2]))))
        y2 = max(0, min(height, int(round(coords[3]))))

        if x2 <= x1 or y2 <= y1:
            return image
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
        Type 2 Visual Question Answering crop-reranking logic.

        Missing media never becomes a fabricated image, and the answer is
        stored together with the candidate/frame identity that produced it.
        This keeps the live route bounded while making VQA evidence
        fail-closed.

        `verify`, when not None, overrides VERIFICATION_RERANK_ENABLED for
        this call - see rerank_type1's docstring for the rationale.
        """
        print(f"Executing Type 2 VQA reranking for query: '{query}'...")
        use_verify = verify if verify is not None else VERIFICATION_RERANK_ENABLED
        questions = self.generate_verification_questions(query) if use_verify else []

        def _score_single_vqa_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
            hit_copy = dict(hit)
            payload = hit_copy["payload"]
            evidence = resolve_candidate_evidence(dataset_dir, payload)
            frame_img = evidence["image"] if evidence is not None else None
            hit_copy["vqa_video_id"] = payload.get("source_file")
            hit_copy["vqa_frame_idx"] = evidence.get("frame_idx") if evidence is not None else None
            hit_copy["vqa_evidence_frame_idx"] = hit_copy["vqa_frame_idx"]
            hit_copy["vqa_evidence_timestamp"] = evidence.get("timestamp") if evidence is not None else None
            hit_copy["vqa_evidence_path"] = evidence.get("path") if evidence is not None else None
            hit_copy["vqa_candidate_id"] = hit_copy.get("id")
            hit_copy["vqa_evidence_available"] = frame_img is not None
            hit_copy["vqa_answer"] = "UNKNOWN"
            hit_copy["vqa_answer_valid"] = False
            hit_copy["vqa_evidence_reason"] = "frame_unavailable" if frame_img is None else ""

            crop_img = frame_img
            if frame_img is not None and self.detector is not None and sub_queries:
                try:
                    detections = self.detector.detect(frame_img, sub_queries)
                except Exception as exc:
                    print(f"Object detection failed for frame: {exc}")
                    detections = []
                if detections:
                    best_det = max(detections, key=lambda x: x.get("conf", 0.0))
                    crop_img = self.crop_bounding_box(frame_img, best_det.get("bbox", []))

            vqa_score = 0.0
            if crop_img is not None:
                vqa_prompt = f"""
Question: {query}
Return ONLY one JSON object with this exact schema:
{{"found": true, "answer": "short answer", "confidence": 0.0, "reason": "..."}}
Use found=false, answer="UNKNOWN", confidence=0.0 when the frame cannot answer the question.
The answer must be grounded in this frame; do not guess.
"""
                try:
                    parsed = parse_grounded_vqa_response(self.vlm.generate(crop_img, vqa_prompt))
                    if parsed["valid"]:
                        vqa_score = parsed["confidence"]
                        hit_copy["vqa_answer"] = parsed["answer"]
                        hit_copy["vqa_answer_valid"] = True
                        hit_copy["vqa_evidence_reason"] = "grounded"
                    else:
                        hit_copy["vqa_evidence_reason"] = parsed["reason"]
                except Exception as exc:
                    print(f"VQA scoring failed for frame: {exc}")
                    hit_copy["vqa_evidence_reason"] = "vlm_error"

            rrf_score = hit_copy.get("rrf_score", 0.0)
            hit_copy["vqa_score"] = vqa_score
            if questions:
                verification_ratio = self.verify_candidate(crop_img, "", questions) if crop_img is not None else 0.0
                hit_copy["verification_ratio"] = verification_ratio
                hit_copy["final_score"] = (
                    TYPE2_RRF_WEIGHT * rrf_score
                    + TYPE2_VQA_WEIGHT * vqa_score
                    + TYPE2_VERIFICATION_WEIGHT * verification_ratio
                )
            else:
                hit_copy["final_score"] = 0.4 * rrf_score + 0.6 * vqa_score
            return hit_copy

        if not candidate_frames:
            return []

        scored = [_score_single_vqa_hit(h) for h in candidate_frames]
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

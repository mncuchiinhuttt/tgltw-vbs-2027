"""
DAKE (Dynamic-Aware Keyframe Extraction) pre-filter - training-free, run
before Adaptive Keyframe Sampling's CLIP-variance pass (see
scene_detector.py's compute_scene_variance/select_diverse_keyframes).

Reference: U-CESE: Unified Clip-based Event Search Engine for AI Challenge
HCMC 2025 (arXiv:2605.23274, team Nomial) - DAKE measures the rate of
change in each frame's JPEG-compressed file size as a free proxy for motion
("dynamic scenes with large motion/texture/lighting changes exhibit abrupt
variations in compressed file size"), with zero model inference. Used here
as a supplementary pre-filter, NOT a replacement for AKS's own CLIP-variance
budget or Qwen-embedding farthest-point sampling - it only cuts the number
of candidate frames those (much more expensive) passes have to process,
before they ever run.
"""
import io
from typing import Any, Dict, List

from PIL import Image


def _jpeg_size(frame_img) -> int:
    """Encodes a frame to JPEG in-memory and returns its byte size - the raw signal DAKE's steepness formula operates on."""
    buf = io.BytesIO()
    Image.fromarray(frame_img).save(buf, format="JPEG", quality=90)
    return buf.tell()


def _steepness(size_i: int, size_j: int, index_gap: int, max_size: float) -> float:
    """
    DAKE's steepness formula (U-CESE arXiv:2605.23274): the normalized rate
    of change in JPEG size between two frames i, j, treated as a slope in
    (index, normalized-size) space - bounded in [0, 1), approaching 1 for a
    large size jump over a short index gap (an abrupt cut/motion event) and
    0 for a small jump over a long gap (a static/slowly-changing span).
    """
    if max_size <= 0:
        return 0.0
    delta = 100.0 * abs(size_j - size_i) / max_size
    denom = (index_gap ** 2 + delta ** 2) ** 0.5
    return delta / denom if denom > 0 else 0.0


def compute_dake_scores(candidate_frames: List[Dict[str, Any]], window: int) -> List[float]:
    """
    Per-candidate aggregated steepness score: for each frame, sums its
    pairwise steepness against every other frame within `window` positions
    on either side (DAKE's own "short temporal windows" - the paper's
    Algorithm 1 uses 3-frame windows, the default here).
    """
    sizes = [_jpeg_size(c["frame_img"]) for c in candidate_frames]
    max_size = float(max(sizes)) if sizes else 0.0

    n = len(sizes)
    scores = []
    for i in range(n):
        total = 0.0
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j == i:
                continue
            total += _steepness(sizes[i], sizes[j], abs(j - i), max_size)
        scores.append(total)
    return scores


def dake_prefilter_candidates(
    candidate_frames: List[Dict[str, Any]],
    keep_ratio: float,
    window: int,
    max_gap: int,
) -> List[Dict[str, Any]]:
    """
    Keeps the top `keep_ratio` fraction of candidate_frames by DAKE steepness
    score, then fills in any gap wider than `max_gap` consecutive dropped
    candidates by keeping the next one anyway - mirrors DAKE's own "at least
    one keyframe every 2x fps frames" safeguard, adapted here to a gap
    measured in candidate-list positions (candidate_frames is already
    downsampled to ~1fps by extract_candidate_frames, not raw video frames)
    rather than a native-fps multiple.

    Returns candidate_frames unchanged if there are too few frames for
    filtering to matter (<= 2), so short scenes are never over-pruned.
    """
    n = len(candidate_frames)
    if n <= 2:
        return candidate_frames

    scores = compute_dake_scores(candidate_frames, window)
    keep_count = max(1, int(round(keep_ratio * n)))

    ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
    kept_indices = set(ranked_indices[:keep_count])

    # Enforce the max-gap safeguard over the ORIGINAL (chronological) order.
    # Adds proactively once the gap REACHES max_gap (>=), not after it's
    # already exceeded (>) - the latter would let the actual resulting gap
    # overshoot max_gap by however far scanning had to look ahead to find
    # the next already-kept index.
    last_kept = -1
    for idx in range(n):
        if idx in kept_indices:
            last_kept = idx
        elif idx - last_kept >= max_gap:
            kept_indices.add(idx)
            last_kept = idx

    return [candidate_frames[i] for i in sorted(kept_indices)]

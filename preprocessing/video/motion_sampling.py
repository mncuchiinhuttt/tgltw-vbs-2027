"""
Fast-pathway frame selection for the Slow/Fast dual-budget scheme.

Slow pathway = existing Adaptive Keyframe Sampling (unchanged - CLIP variance
decides a 1-8 frame budget, farthest-point sampling picks them in Qwen
embedding space). Lives in scene_detector.py already.

Fast pathway = this module. Independent of Slow: runs on the scene's raw
decoded frames (before AKS filters anything), measures real pixel-level
motion via optical flow (cheap, no model forward pass), and returns more
frames than Slow but concentrated where motion is actually happening -
rather than either (a) uniform sampling, which SlowFast-LLaVA's own authors
note can miss short bursts of motion, or (b) reusing CLIP embedding variance,
which is a semantic/global proxy that can miss localized motion (e.g. a small
object moving fast against a large static background) - see the accompanying
report for the papers behind this reasoning (Flow4Agent arXiv:2510.05836,
KITE arXiv:2604.07034).
"""
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def compute_motion_energy(frames: Sequence[np.ndarray]) -> np.ndarray:
    """
    frames: RGB frames (matching scene_detector.py's extract_candidate_frames,
    which already converts BGR->RGB before returning "frame_img"), in
    temporal order, at whatever fps they were decoded at.

    Returns: float array of length len(frames) - 1, where element i is the
    mean optical-flow magnitude between frames[i] and frames[i+1] - i.e. how
    much genuinely moved between those two instants. Returns an empty array
    if fewer than 2 frames are given (nothing to compare).
    """
    if len(frames) < 2:
        return np.zeros(0, dtype=np.float32)

    energies = np.zeros(len(frames) - 1, dtype=np.float32)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        energies[i - 1] = float(np.linalg.norm(flow, axis=2).mean())
        prev_gray = curr_gray
    return energies


def compute_fast_pathway_frame_count(
    scene_duration_sec: float,
    fps_target: float,
    min_frames: int,
    max_frames: int,
) -> int:
    """
    How many Fast-pathway frames a scene of this length deserves - scaled by
    duration rather than a fixed constant (SF-LLaVA's "50 frames" is a
    whole-video number; scenes here are usually a few seconds, so a fixed 50
    would mean sampling nearly every decoded frame). Clamped to
    [min_frames, max_frames] so very short scenes still get a minimum amount
    of temporal coverage and very long scenes don't blow the token budget.
    """
    raw = round(scene_duration_sec * fps_target)
    return int(min(max(raw, min_frames), max_frames))


def sample_timestamps_by_motion(
    motion_energy: np.ndarray,
    timestamps: np.ndarray,
    num_samples: int,
    static_floor_ratio: float = 0.05,
) -> np.ndarray:
    """
    Inverse-CDF (inverse transform) sampling over motion energy: treats
    motion_energy as an unnormalized density over time and draws
    `num_samples` timestamps from it, so segments with more motion naturally
    receive more samples and static segments receive fewer (but not zero).

    motion_energy: array of length N-1 (one value per interval between
        consecutive decoded frames - see compute_motion_energy).
    timestamps: array of length N (timestamps[i] is the time of frame i;
        motion_energy[i] describes the interval [timestamps[i], timestamps[i+1]]).
    num_samples: how many timestamps to return.
    static_floor_ratio: a small floor (as a fraction of the mean energy)
        added to every interval so a scene that's motionless everywhere
        doesn't collapse into duplicate samples at a single spot, and a scene
        that's static in most places but has one motion spike still spares a
        few samples for the static parts (useful in case the motion signal
        itself is noisy/wrong in a spot).

    Returns: sorted array of `num_samples` timestamps (may contain values
    between two decoded frames - snap to the nearest actual frame when
    extracting pixels, see select_fast_pathway_frames below).
    """
    if len(motion_energy) == 0 or num_samples <= 0:
        idx = np.linspace(0, len(timestamps) - 1, max(num_samples, 1)).round().astype(int)
        return timestamps[idx]

    mean_energy = motion_energy.mean()
    floor = static_floor_ratio * mean_energy if mean_energy > 0 else 1.0
    weights = np.clip(motion_energy + floor, a_min=1e-8, a_max=None)

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    quantiles = (np.arange(num_samples) + 0.5) / num_samples
    segment_idx = np.clip(np.searchsorted(cdf, quantiles), 0, len(weights) - 1)

    sampled = np.empty(num_samples, dtype=np.float64)
    for i, (q, seg) in enumerate(zip(quantiles, segment_idx)):
        seg_start_cdf = cdf[seg - 1] if seg > 0 else 0.0
        seg_span = max(cdf[seg] - seg_start_cdf, 1e-8)
        frac = np.clip((q - seg_start_cdf) / seg_span, 0.0, 1.0)
        sampled[i] = timestamps[seg] + frac * (timestamps[seg + 1] - timestamps[seg])

    return np.sort(sampled)


def select_fast_pathway_frames(
    frames: Sequence[np.ndarray],
    frame_timestamps: Sequence[float],
    scene_duration_sec: float,
    fps_target: float,
    min_frames: int,
    max_frames: int,
) -> List[Tuple[float, np.ndarray]]:
    """
    Entry point scene_detector.py should call. Given ALL raw decoded frames
    of a scene (not the AKS-filtered Slow keyframes - this runs independently
    and in parallel to AKS, not after it), returns a motion-weighted subset
    as (timestamp, frame) pairs, ready to be resized down to the Fast
    pathway's low pixel budget and sent to the VLM alongside the Slow frames.
    """
    num_fast_frames = compute_fast_pathway_frame_count(scene_duration_sec, fps_target, min_frames, max_frames)

    if len(frames) <= num_fast_frames:
        # Scene too short/sparsely decoded to subsample further - use everything.
        return list(zip(frame_timestamps, frames))

    motion_energy = compute_motion_energy(frames)
    timestamps_arr = np.asarray(frame_timestamps, dtype=np.float64)
    sampled_ts = sample_timestamps_by_motion(motion_energy, timestamps_arr, num_fast_frames)

    result = []
    seen_indices = set()
    for t in sampled_ts:
        idx = int(np.argmin(np.abs(timestamps_arr - t)))
        if idx in seen_indices:
            continue  # avoid duplicate frames if two samples snap to the same index
        seen_indices.add(idx)
        result.append((frame_timestamps[idx], frames[idx]))

    return sorted(result, key=lambda pair: pair[0])
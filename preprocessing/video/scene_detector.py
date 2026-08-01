import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from scenedetect import detect, ContentDetector
from preprocessing.config import (
    SCENE_DETECTION_THRESHOLD, KEYFRAME_VARIANCE_LOW, KEYFRAME_VARIANCE_MID, KEYFRAME_MAX_BUDGET,
    FAST_PATHWAY_DENSE_SAMPLING_FPS, FAST_PATHWAY_FPS_TARGET,
    FAST_PATHWAY_MIN_FRAMES, FAST_PATHWAY_MAX_FRAMES,
    SCENE_MERGE_WINDOW_SEC, SCENE_MERGE_SAMPLES_PER_SIDE, SCENE_MERGE_THRESHOLD_RATIO,
)
from preprocessing.video.motion_sampling import select_fast_pathway_frames

def detect_scenes(video_path: str, threshold: float = SCENE_DETECTION_THRESHOLD) -> List[Tuple[float, float]]:
    """
    Detect scene boundaries in a video using PySceneDetect.
    Returns:
        List of tuples representing (start_time_seconds, end_time_seconds) for each scene.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    print(f"Detecting scenes in video: {video_path} ({frame_count:.0f} frames, {duration:.1f}s @ {fps:.1f}fps)...")
    scene_list = detect(video_path, ContentDetector(threshold=threshold), show_progress=True)
    scenes = []
    for scene in scene_list:
        start_sec = scene[0].get_seconds()
        end_sec = scene[1].get_seconds()
        scenes.append((start_sec, end_sec))
        
    if not scenes:
        # Fallback: treat the entire video as a single scene
        if duration <= 0:
            duration = 10.0

        scenes.append((0.0, duration))
        print(f"No scene cuts detected. Treating entire video as a single scene (0.0s - {duration:.2f}s).")
    else:
        print(f"Detected {len(scenes)} scenes.")
        
    return scenes

def extract_candidate_frames(video_path: str, start_sec: float, end_sec: float, sampling_rate_fps: float = 1.0) -> List[Dict[str, Any]]:
    """
    Extract candidate frames from a scene for diversity sampling.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    
    # Calculate step size based on sampling rate
    step = max(1, int(fps / sampling_rate_fps))
    
    candidate_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame_idx = start_frame
    while current_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if (current_frame_idx - start_frame) % step == 0:
            timestamp = current_frame_idx / fps
            # Convert BGR (OpenCV) to RGB (standard PIL/numpy format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            candidate_frames.append({
                "frame_img": frame_rgb,
                "timestamp": timestamp,
                "frame_idx": current_frame_idx
            })
            
        current_frame_idx += 1
    
    cap.release()
    return candidate_frames

def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def _avg_pairwise_sim(embeds: List[np.ndarray]) -> float:
    """Average cosine similarity across all pairs in embeds; 1.0 if fewer than 2 (nothing to compare)."""
    if len(embeds) < 2:
        return 1.0
    sims = [cosine_sim(embeds[i], embeds[j]) for i in range(len(embeds)) for j in range(i + 1, len(embeds))]
    return sum(sims) / len(sims)

def _avg_cross_sim(embeds_a: List[np.ndarray], embeds_b: List[np.ndarray]) -> float:
    """Average cosine similarity between every pair (a, b) across the two groups; 0.0 if either is empty."""
    if not embeds_a or not embeds_b:
        return 0.0
    sims = [cosine_sim(a, b) for a in embeds_a for b in embeds_b]
    return sum(sims) / len(sims)

def _boundary_is_weak(
    before_embeds: List[np.ndarray],
    after_embeds: List[np.ndarray],
    threshold_ratio: float = SCENE_MERGE_THRESHOLD_RATIO,
) -> bool:
    """
    VIREO-inspired (VBS2026, MMM 2026 LNCS 16415 ch.19) "merging weak
    boundaries": compares average within-scene similarity on each side of a
    candidate cut to the cross-boundary similarity between them. If the cut
    barely separates anything relative to how much each side already
    varies internally - cross-boundary similarity is at least
    threshold_ratio of the average within-scene similarity - the boundary
    is "weak" and should be merged away.
    """
    if not before_embeds or not after_embeds:
        return False
    within_avg = (_avg_pairwise_sim(before_embeds) + _avg_pairwise_sim(after_embeds)) / 2.0
    if within_avg <= 0:
        return False
    cross = _avg_cross_sim(before_embeds, after_embeds)
    return (cross / within_avg) >= threshold_ratio

def refine_scene_boundaries(
    video_path: str,
    scenes: List[Tuple[float, float]],
    clip_embedder,
    window_sec: float = SCENE_MERGE_WINDOW_SEC,
    samples_per_side: int = SCENE_MERGE_SAMPLES_PER_SIDE,
    threshold_ratio: float = SCENE_MERGE_THRESHOLD_RATIO,
) -> List[Tuple[float, float]]:
    """
    VIREO-inspired (VBS2026) scoped-down "merging weak boundaries"
    post-process on PySceneDetect's own candidate cuts - see
    SCENE_MERGE_ENABLED in config.py for why this doesn't reimplement
    VIREO's full self-similarity-matrix + kernel-temporal-segmentation
    pipeline. Samples a few frames on each side of every adjacent scene
    pair, embeds them with the already-loaded LightweightCLIPEmbedder
    (compute_scene_variance's embedder, not a separate BLIP load), and
    merges the pair when the cut is "weak" (_boundary_is_weak). Repeats
    until no more merges happen in a full pass - a merge can make a
    previously-strong neighboring boundary weak too, e.g. three
    near-identical short shots in a row.
    """
    if len(scenes) < 2:
        return scenes

    merged = list(scenes)
    changed = True
    while changed and len(merged) >= 2:
        changed = False
        i = 0
        while i < len(merged) - 1:
            start_i, _ = merged[i]
            _, end_next = merged[i + 1]
            boundary_t = merged[i][1]  # == merged[i+1][0] for PySceneDetect's contiguous scene list

            sampling_fps = max(1.0, samples_per_side / window_sec) if window_sec > 0 else 1.0
            before = extract_candidate_frames(
                video_path, max(start_i, boundary_t - window_sec), boundary_t,
                sampling_rate_fps=sampling_fps,
            )[-samples_per_side:]
            after = extract_candidate_frames(
                video_path, boundary_t, min(end_next, boundary_t + window_sec),
                sampling_rate_fps=sampling_fps,
            )[:samples_per_side]

            before_embeds = [clip_embedder.embed_image(f["frame_img"]) for f in before]
            after_embeds = [clip_embedder.embed_image(f["frame_img"]) for f in after]

            if _boundary_is_weak(before_embeds, after_embeds, threshold_ratio):
                merged[i] = (start_i, end_next)
                del merged[i + 1]
                changed = True
                # stay at i to re-check the newly-merged scene against its new next neighbor
            else:
                i += 1

    return merged

def compute_scene_variance(candidate_frames: List[Dict[str, Any]], clip_embedder) -> float:
    """
    Adaptive Keyframe Sampling, step 1: encode all of a scene's candidate
    frames with a lightweight CLIP model and measure how much they vary.
    A low variance means a "static" scene (e.g. a talking head); a high
    variance means a "dynamic" scene with lots of visual change - this
    drives how many keyframes get kept (see get_adaptive_budget).
    """
    if not candidate_frames:
        return 0.0
    embeds = np.stack([clip_embedder.embed_image(f["frame_img"]) for f in candidate_frames])
    return float(np.var(embeds, axis=0).mean())

def get_adaptive_budget(variance: float) -> int:
    """
    Adaptive Keyframe Sampling, step 2: map a scene's variance to how many
    keyframes to keep for it (capped at KEYFRAME_MAX_BUDGET).
    """
    if variance < KEYFRAME_VARIANCE_LOW:
        return 1
    elif variance < KEYFRAME_VARIANCE_MID:
        return 2
    return min(KEYFRAME_MAX_BUDGET, max(1, int(variance * 100)))

def select_diverse_keyframes(
    candidate_frames: List[Dict[str, Any]],
    embedder,
    budget: int
) -> List[Dict[str, Any]]:
    """
    Adaptive Keyframe Sampling, step 3: farthest-point sampling down to
    `budget` frames, using Qwen3-Embedding-VL-8B (passed as `embedder`) as
    the distance space - at each step, keep whichever remaining candidate is
    least similar to everything already selected, maximizing coverage of
    the scene's content within the budget.
    """
    if not candidate_frames:
        return []

    print(f"  Embedding {len(candidate_frames)} candidate frames for diversity sampling...")
    for i, frame in enumerate(candidate_frames, start=1):
        print(f"  Embedding frame {i}/{len(candidate_frames)} (t={frame['timestamp']:.2f}s)...")
        frame["embed"] = embedder.embed_image(frame["frame_img"])

    if len(candidate_frames) <= budget:
        return candidate_frames

    selected = [candidate_frames[0]]
    remaining = candidate_frames[1:]

    while len(selected) < budget and remaining:
        best_idx, best_dist = -1, -1.0
        for i, frame in enumerate(remaining):
            sims = [cosine_sim(frame["embed"], s["embed"]) for s in selected]
            dist = 1 - max(sims)
            if dist > best_dist:
                best_dist, best_idx = dist, i
        # Remove by index rather than list.remove(): remaining holds dicts
        # with numpy-array "embed" values, and list.remove() uses == equality,
        # which raises on dicts containing arrays ("truth value of an array
        # with more than one element is ambiguous").
        selected.append(remaining.pop(best_idx))

    return selected

def select_fast_pathway_frames_for_scene(
    video_path: str,
    start_sec: float,
    end_sec: float,
    dense_sampling_fps: float = FAST_PATHWAY_DENSE_SAMPLING_FPS,
    fps_target: float = FAST_PATHWAY_FPS_TARGET,
    min_frames: int = FAST_PATHWAY_MIN_FRAMES,
    max_frames: int = FAST_PATHWAY_MAX_FRAMES,
) -> List[Dict[str, Any]]:
    """
    Fast pathway: independent of AKS/Slow above (does not read
    diverse_keyframes or its budget) - decodes this scene at a denser fps
    than AKS's diversity candidates need (motion needs finer temporal
    resolution than farthest-point sampling does), measures real
    optical-flow motion between consecutive decoded frames, and returns a
    motion-weighted subset via inverse-CDF sampling - concentrated where the
    scene actually moves, rather than spread uniformly or missed entirely by
    a variance-based proxy. Reuses extract_candidate_frames (just at a
    higher sampling_rate_fps) rather than a separate decode path, and
    returns the same {"frame_img", "timestamp", "frame_idx"} shape AKS's own
    candidates/keyframes use, so callers (main.py) handle both identically
    (e.g. the same `Image.fromarray(f["frame_img"])` conversion).

    Cost note: this decodes each scene a second time (~dense_sampling_fps,
    default 8fps) on top of AKS's own 1fps candidate decode above - roughly
    8x more per-scene video I/O for the Fast pathway alone. Intentional
    trade of CPU decode/optical-flow cost for VLM token savings; worth
    measuring actual end-to-end preprocessing throughput at dataset scale
    rather than assuming it's a net win.
    """
    dense_frames = extract_candidate_frames(video_path, start_sec, end_sec, sampling_rate_fps=dense_sampling_fps)
    if not dense_frames:
        return []

    frame_imgs = [f["frame_img"] for f in dense_frames]
    timestamps = [f["timestamp"] for f in dense_frames]
    scene_duration = end_sec - start_sec

    sampled = select_fast_pathway_frames(
        frame_imgs, timestamps, scene_duration_sec=scene_duration,
        fps_target=fps_target, min_frames=min_frames, max_frames=max_frames,
    )

    return [{"frame_img": frame, "timestamp": ts, "frame_idx": None} for ts, frame in sampled]
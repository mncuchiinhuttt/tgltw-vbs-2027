import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from scenedetect import detect, ContentDetector
from preprocessing.config import (
    SCENE_DETECTION_THRESHOLD, KEYFRAME_VARIANCE_LOW, KEYFRAME_VARIANCE_MID, KEYFRAME_MAX_BUDGET
)

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

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from scenedetect import detect, ContentDetector
from preprocessing.config import SCENE_DETECTION_THRESHOLD, KEYFRAME_DIVERSITY_THRESHOLD

def detect_scenes(video_path: str, threshold: float = SCENE_DETECTION_THRESHOLD) -> List[Tuple[float, float]]:
    """
    Detect scene boundaries in a video using PySceneDetect.
    Returns:
        List of tuples representing (start_time_seconds, end_time_seconds) for each scene.
    """
    print(f"Detecting scenes in video: {video_path}...")
    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    scenes = []
    for scene in scene_list:
        start_sec = scene[0].get_seconds()
        end_sec = scene[1].get_seconds()
        scenes.append((start_sec, end_sec))
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

def select_diverse_keyframes(
    candidate_frames: List[Dict[str, Any]], 
    embedder, 
    threshold: float = KEYFRAME_DIVERSITY_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Filter candidate frames in a scene using Diversity Sampling.
    Uses Qwen3-Embedding-VL-8B (passed as 'embedder') to generate frame embeddings.
    """
    if not candidate_frames:
        return []
        
    # Generate embedding for candidate 0
    candidate_frames[0]["embed"] = embedder.embed_image(candidate_frames[0]["frame_img"])
    selected = [candidate_frames[0]]
    
    for frame in candidate_frames[1:]:
        frame["embed"] = embedder.embed_image(frame["frame_img"])
        sims = [cosine_sim(frame["embed"], s["embed"]) for s in selected]
        
        # If the frame is different enough from all previously selected frames, keep it
        if max(sims) < (1 - threshold):
            selected.append(frame)
            
    return selected

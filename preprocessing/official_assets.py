"""
Loaders for the official AIC dataset assets BTC (organizers) provide
alongside raw videos (see Thong tin vong So tuyen AIC2026.pdf, section 3):
pre-extracted Keyframes, Objects (Faster R-CNN/OpenImages V4 detections),
CLIP features (clip-ViT-B-32), and Metadata (YouTube JSON).

These are optional supplementary sources - preprocessing/main.py falls back
entirely to its own from-scratch pipeline (YOLOE+SAM3 detection,
LightweightCLIPEmbedder variance estimation) whenever a given asset isn't
found on disk for a video/keyframe, so this has zero effect on datasets that
don't ship these files.

!! VERIFY AGAINST REAL DATA BEFORE TRUSTING !!: the PDF describes these
assets in prose, not a schema - no real downloaded sample was available when
this was written. Field/path names below are reconstructed best-effort:
  - load_official_objects's TF Object Detection API JSON schema (the PDF
    just links to TensorFlow's own docs, not a schema itself).
  - Whether the CLIP-features .npy is genuinely one-per-video (assumed here,
    mirroring every other asset type's per-video structure) or one giant
    file for the whole corpus (the PDF's Vietnamese wording "một file .npy
    duy nhất" is ambiguous between the two).
  - load_official_keyframe_index_map's file path/schema - the PDF says a
    keyframe's native frame index "được ghi trong file metadata" but never
    names that file. Kept in its own function so it's the one thing to
    correct first once a real sample is in hand - everything else here
    depends on it to map our own extracted keyframes to BTC's.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _video_stem(video_name: str) -> str:
    """'L01_V001.mp4' -> 'L01_V001' - the stem BTC's per-video folders/files are keyed by."""
    return Path(video_name).stem


def load_official_keyframe_index_map(dataset_dir: str, video_name: str) -> Dict[str, int]:
    """
    Loads BTC's keyframe-filename -> native-video-frame-index mapping for
    one video, from the (assumed) Keyframes/<video_stem>/frame_index.json,
    shaped {"0000.jpg": 120, "0001.jpg": 340, ...}. This is the single least
    certain assumption in this module (see module docstring) - returns {}
    if the file doesn't exist, so callers can gracefully skip Objects/CLIP
    augmentation for videos without it rather than guessing wrong.
    """
    path = Path(dataset_dir) / "Keyframes" / _video_stem(video_name) / "frame_index.json"
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to parse official keyframe index map {path}: {e}")
        return {}


def nearest_official_keyframe(
    keyframe_index_map: Dict[str, int], target_frame_idx: int, max_distance: int = 5,
) -> Optional[str]:
    """
    Finds the BTC keyframe filename whose native frame index is closest to
    target_frame_idx (our own extracted keyframe's frame_idx) - our own
    scene-detection + Adaptive Keyframe Sampling picks different frames than
    BTC's own keyframe extraction, so there's rarely an exact frame_idx
    match, only a nearby one. Returns None if the map is empty or nothing
    is within max_distance frames.
    """
    if not keyframe_index_map:
        return None

    best_filename, best_distance = None, max_distance + 1
    for filename, frame_idx in keyframe_index_map.items():
        distance = abs(frame_idx - target_frame_idx)
        if distance < best_distance:
            best_filename, best_distance = filename, distance

    return best_filename if best_distance <= max_distance else None


def load_official_objects(dataset_dir: str, video_name: str, keyframe_filename: str) -> List[Dict[str, Any]]:
    """
    Loads one keyframe's Faster R-CNN (OpenImages V4) detections from
    Objects/<video_stem>/<keyframe_stem>.json - filename mirrors the
    keyframe's own filename per the PDF (keyframe L01_V001/0000.jpg ->
    Objects/L01_V001/0000.json).

    Assumed TF Object Detection API export schema:
    {"detection_boxes": [[y1,x1,y2,x2], ...] (normalized 0-1),
     "detection_classes": [...], "detection_scores": [...],
     "detection_class_names": [...] (optional, human-readable label strings)}

    Returns a list shaped like ObjectDetector.detect()'s own output
    ({"label", "bbox", "conf"}) so it can be merged directly into
    detected_objects alongside YOLOE+SAM3 detections - bbox stays
    NORMALIZED (0-1), not pixel coordinates, since this function has no
    image to denormalize against; callers must multiply by the keyframe's
    actual (width, height) before merging/IoU-deduping against pixel-space
    detections. Returns [] if no matching file exists.
    """
    keyframe_stem = Path(keyframe_filename).stem
    json_path = Path(dataset_dir) / "Objects" / _video_stem(video_name) / f"{keyframe_stem}.json"
    if not json_path.exists():
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to parse official Objects file {json_path}: {e}")
        return []

    boxes = data.get("detection_boxes", [])
    classes = data.get("detection_class_names", data.get("detection_classes", []))
    scores = data.get("detection_scores", [])

    detections = []
    for box, cls, score in zip(boxes, classes, scores):
        if len(box) != 4:
            continue
        y1, x1, y2, x2 = box
        detections.append({
            "label": str(cls),
            "bbox": [x1, y1, x2, y2],
            "conf": float(score),
            "source": "official_objects",
        })
    return detections


def load_official_clip_feature(dataset_dir: str, video_name: str, keyframe_filename: str) -> Optional[np.ndarray]:
    """
    Loads one keyframe's precomputed clip-ViT-B-32 feature vector from the
    per-video CLIP features .npy file (assumed path:
    clip-features-vit-b32/<video_stem>.npy, shape (num_keyframes, 512),
    row order matching Keyframes/<video_stem>/'s own sorted filename order
    per the PDF). Returns None if the file/index doesn't exist.
    """
    npy_path = Path(dataset_dir) / "clip-features-vit-b32" / f"{_video_stem(video_name)}.npy"
    if not npy_path.exists():
        return None

    keyframes_dir = Path(dataset_dir) / "Keyframes" / _video_stem(video_name)
    if not keyframes_dir.exists():
        return None
    sorted_filenames = sorted(p.name for p in keyframes_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if keyframe_filename not in sorted_filenames:
        return None
    row_index = sorted_filenames.index(keyframe_filename)

    try:
        features = np.load(npy_path)
    except (OSError, ValueError) as e:
        print(f"Warning: failed to load official CLIP features {npy_path}: {e}")
        return None

    if row_index >= len(features):
        return None
    return features[row_index]


def load_official_metadata(dataset_dir: str, video_name: str) -> Dict[str, Any]:
    """
    Loads a video's YouTube metadata JSON (Metadata/<video_stem>.json per
    the PDF) - title/description/etc, used as extra BM25 signal. Returns {}
    if no metadata file exists for this video (the PDF notes some videos in
    the provided data may not have a corresponding metadata file).
    """
    json_path = Path(dataset_dir) / "Metadata" / f"{_video_stem(video_name)}.json"
    if not json_path.exists():
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to parse official Metadata file {json_path}: {e}")
        return {}

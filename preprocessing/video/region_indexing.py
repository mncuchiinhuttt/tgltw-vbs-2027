"""
Region-level index points built from the SAM3 proposals a keyframe already
produced for detection and OCR gating.

A single pooled frame embedding is an average over the whole image, so a
license plate, a shop sign, or one small object in a wide shot contributes a
fraction of the vector and is effectively unsearchable.  Embedding the crop
separately gives that content its own point.  The regions cost nothing extra
to obtain - RegionProposer.propose already ran to decide whether detection and
OCR should happen at all - so the added cost is embedding calls, not a model.

Region points deliberately use `modality="region"`, never `"visual"`.  The
retrieval side keys three mechanisms off frame identity: temporal coherence
boosting sums the scores of other candidates within N frames, scene
diversification keeps one hit per (video, scene), and TRAKE aligns events
along a video's frame timeline.  Crops share their parent frame's index, so
admitting them as ordinary visual points would let a frame with many regions
boost itself, let a crop evict its own parent from the result grid, and fill
TRAKE's timeline with duplicates.  Keeping them in a separate modality means
they are only ever seen by code that asked for them.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence

import cv2
from PIL import Image


def probe_video_fps(video_path: str, default: float = 25.0) -> float:
    """Frame rate of a video, or `default` when it cannot be read.

    Needed to turn shot timestamps into the native frame indices the
    submission format is scored on, for corpora whose shot boundaries are
    given only in seconds.
    """
    capture = cv2.VideoCapture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    return fps if fps > 0 else default


def stable_region_point_id(video_name: str, parent_point_id: str, region_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbs-region:{video_name}:{parent_point_id}:{region_index}"))


def select_regions(
    regions: Sequence[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    max_regions: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> List[Dict[str, Any]]:
    """
    Keep the highest-scoring proposals that are worth their own embedding.

    Regions covering almost the whole frame are dropped because their crop is
    a near-duplicate of the parent frame's own embedding, and specks are
    dropped because upscaled noise embeds as nothing in particular.
    """
    frame_area = float(frame_width * frame_height)
    if frame_area <= 0:
        return []

    usable = []
    for region in regions:
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width, height = x2 - x1, y2 - y1
        if width <= 1 or height <= 1:
            continue
        ratio = (width * height) / frame_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        usable.append(region)

    usable.sort(key=lambda region: float(region.get("score", 0.0)), reverse=True)
    return usable[:max_regions]


def crop_region(frame_img: Image.Image, bbox: Sequence[float]) -> Optional[Image.Image]:
    """Crop a proposal out of its parent frame, clamped to the frame."""
    width, height = frame_img.size
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(width, int(bbox[2]))
    y2 = min(height, int(bbox[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame_img.crop((x1, y1, x2, y2))


def index_region_crops(
    *,
    indexer,
    embedder,
    video_name: str,
    frame_img: Image.Image,
    regions: Sequence[Dict[str, Any]],
    parent_point_id: str,
    parent_payload: Dict[str, Any],
    max_regions: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> int:
    """Embed and index the selected crops of one keyframe.  Returns the count."""
    width, height = frame_img.size
    chosen = select_regions(regions, width, height, max_regions, min_area_ratio, max_area_ratio)
    if not chosen:
        return 0

    indexed = 0
    for region_index, region in enumerate(chosen):
        crop = crop_region(frame_img, region["bbox"])
        if crop is None:
            continue
        try:
            vector = embedder.embed_image(crop)
        except Exception as exc:  # a single bad crop must not abort the video
            print(f"  Warning: region embedding failed ({exc}); skipping region {region_index}.")
            continue

        concept = str(region.get("concept", "")).strip()
        payload = {
            "modality": "region",
            "source_file": video_name,
            "parent_point_id": parent_point_id,
            "timestamp": parent_payload.get("timestamp"),
            "frame_idx": parent_payload.get("frame_idx"),
            "scene_id": parent_payload.get("scene_id"),
            "shot_id": parent_payload.get("shot_id"),
            "region_bbox": [float(value) for value in region["bbox"]],
            "region_concept": concept,
            "region_score": float(region.get("score", 0.0)),
            "text_blob": " . ".join(filter(None, [concept, parent_payload.get("ocr_text", "")])),
        }
        indexer.index_visual_point(
            stable_region_point_id(video_name, parent_point_id, region_index), vector, payload
        )
        indexed += 1
    return indexed

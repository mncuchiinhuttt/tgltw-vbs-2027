"""
Decode frames that were never indexed, for the handful of videos a query has
already narrowed down to.

This is not re-ranking.  Re-ranking reorders what the index already contains,
so it can never recover a moment offline selection dropped - if no frame of
that moment was ever embedded, no amount of scoring will surface it.  Once the
first pass has identified a few candidate videos, decoding those videos
directly and scoring fresh frames against the query lifts that ceiling: the
frames are created in response to the query rather than chosen in advance.

The cost is bounded by construction - only the top few videos are touched, and
only up to a fixed frame count each - but it does require the raw videos to be
readable from the query host, which is why the caller keeps it behind a flag.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def resolve_video_path(video_source_dir: str, video_name: str) -> Optional[str]:
    """Locate a video by the `source_file` recorded in its index payloads."""
    if not video_source_dir or not video_name:
        return None
    root = Path(video_source_dir).expanduser()
    direct = root / video_name
    if direct.is_file():
        return str(direct)
    matches = [path for path in root.rglob(video_name) if path.is_file()]
    return str(matches[0]) if matches else None


def decode_frames(video_path: str, sampling_fps: float, max_frames: int) -> List[Dict[str, Any]]:
    """Sample a whole video at `sampling_fps`, capped at `max_frames`."""
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / sampling_fps))) if sampling_fps > 0 else 1
        if total > 0 and max_frames > 0:
            # Spread the budget across the whole video rather than exhausting
            # it on the opening seconds.
            step = max(step, total // max_frames or 1)

        frames: List[Dict[str, Any]] = []
        index = 0
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                frames.append({
                    "frame_img": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    "frame_idx": index,
                    "timestamp": index / fps,
                })
            index += 1
        return frames
    finally:
        capture.release()


def extract_query_time_frames(
    *,
    video_name: str,
    video_source_dir: str,
    embedder,
    query_vector: np.ndarray,
    known_frame_indices: set,
    sampling_fps: float,
    max_frames: int,
    top_frames: int,
) -> List[Dict[str, Any]]:
    """
    Return the best-scoring freshly decoded frames of one video.

    Frames whose native index is already indexed are skipped - those are
    in_video_refine's job, and re-embedding them would only duplicate points
    the caller already has.  Each result carries a `query_time` marker and no
    point id, so callers can tell a decoded frame from a stored one.
    """
    video_path = resolve_video_path(video_source_dir, video_name)
    if video_path is None or not os.path.isfile(video_path):
        return []

    query = np.asarray(query_vector, dtype=np.float64).ravel()
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return []

    scored = []
    for frame in decode_frames(video_path, sampling_fps, max_frames):
        if frame["frame_idx"] in known_frame_indices:
            continue
        try:
            vector = np.asarray(embedder.embed_image(frame["frame_img"]), dtype=np.float64).ravel()
        except Exception as exc:
            print(f"Warning: query-time embedding failed for {video_name} ({exc}).")
            return []
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm == 0.0:
            continue
        similarity = float(np.dot(query, vector) / (query_norm * vector_norm))
        scored.append((similarity, frame))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "similarity": similarity,
            "payload": {
                "modality": "visual",
                "source_file": video_name,
                "timestamp": frame["timestamp"],
                "frame_idx": frame["frame_idx"],
                "query_time": True,
                "caption": "",
                "text_blob": "",
            },
        }
        for similarity, frame in scored[:top_frames]
    ]

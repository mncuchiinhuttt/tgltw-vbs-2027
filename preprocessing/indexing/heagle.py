"""Pure helpers for the H-EAGLE-lite shot-level index."""

from __future__ import annotations

import uuid
from typing import Any, Iterable

import numpy as np


def stable_shot_id(video_name: str, scene_idx: int, official_shot_id: str | None = None) -> str:
    """Return an auditable shot id that is stable across resume runs."""
    return f"{video_name}:{official_shot_id}" if official_shot_id else f"{video_name}:shot:{scene_idx:06d}"


def stable_shot_point_id(video_name: str, shot_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbs-shot:{video_name}:{shot_id}"))


def stable_frame_point_id(video_name: str, frame_key: str) -> str:
    """Stable frame point ID even when a detector changes shot boundaries."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbs-frame:{video_name}:{frame_key}"))


def aggregate_shot_embedding(
    frame_vectors: Iterable[np.ndarray],
    quality_scores: Iterable[float | None] | None = None,
) -> np.ndarray:
    """Mean-pool representative frame vectors and L2-normalize the result.

    Sharpness is deliberately only a small positive weight.  Semantic frame
    coverage remains the dominant signal, so a visually sharp but redundant
    frame cannot erase the rest of a shot's content.
    """
    vectors = [np.asarray(vector, dtype=np.float32) for vector in frame_vectors]
    if not vectors:
        raise ValueError("cannot aggregate an empty shot")
    matrix = np.stack(vectors)
    if quality_scores is None:
        weights = np.ones(len(vectors), dtype=np.float32)
    else:
        scores = [0.0 if value is None else float(value) for value in quality_scores]
        if len(scores) != len(vectors):
            raise ValueError("quality_scores must have the same length as frame_vectors")
        weights = 0.75 + 0.25 * np.clip(np.asarray(scores, dtype=np.float32), 0.0, 1.0)
    result = np.average(matrix, axis=0, weights=weights)
    norm = float(np.linalg.norm(result))
    if norm == 0.0:
        return result
    return result / norm


def shot_payload(
    *,
    video_name: str,
    shot_id: str,
    scene_idx: int,
    start_sec: float,
    end_sec: float,
    frame_point_ids: list[str],
    frame_timestamps: list[float],
    frame_count: int,
    text_blob: str,
) -> dict[str, Any]:
    return {
        "modality": "shot",
        "source_file": video_name,
        "shot_id": shot_id,
        "scene_id": scene_idx,
        "timestamp": start_sec,
        "timestamp_end": end_sec,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "representative_frame_ids": frame_point_ids,
        "representative_timestamps": frame_timestamps,
        "frame_count": frame_count,
        "text_blob": text_blob,
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Corpus Indexer:
Quét và index toàn bộ 1.703 video MP4 đã tải về máy vào Qdrant (WeMM-4B).
"""

import os
import sys
import time
import json
import uuid
import cv2
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
INFERENCE_DIR = REPO_ROOT / "inference-code"
DATASET_DIR = REPO_ROOT / "datasets" / "v3c"
VIDEO_DIR = DATASET_DIR / "videos"
METADATA_DIR = DATASET_DIR / "metadata"
KEYFRAME_DIR = DATASET_DIR / "keyframes"
LOG_DIR = REPO_ROOT / "evaluation" / "indexing_logs"

for p in (str(REPO_ROOT), str(INFERENCE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
    VISUAL_COLLECTION_NAME, EMBEDDING_MRL_DIM,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from models.embedding import WeMMEmbedding4BEmbedder


def extract_video_keyframes(video_path: Path, sample_interval_sec: float = 3.0, max_frames: int = 15) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_step = max(1, int(fps * sample_interval_sec))
    keyframes = []

    video_id = video_path.stem
    out_dir = KEYFRAME_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    for frame_idx in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        ts = frame_idx / fps
        save_path = out_dir / f"{frame_idx}.jpg"
        if not save_path.exists():
            cv2.imwrite(str(save_path), frame)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keyframes.append({
            "source_file": video_path.name,
            "frame_idx": frame_idx,
            "timestamp": ts,
            "keyframe_path": str(save_path),
            "image": img_rgb,
        })
        frame_count += 1
        if frame_count >= max_frames:
            break

    cap.release()
    return keyframes


def load_video_metadata(video_id: str) -> Dict[str, Any]:
    info_path = METADATA_DIR / "info" / f"{video_id}.json"
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"title": f"V3C Video {video_id}", "description": "", "tags": []}


def main():
    print("=== STARTING FULL CORPUS 1,703 VIDEOS INDEXING ===")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY or None)
    visual_dim = EMBEDDING_MRL_DIM or 2048

    if not client.collection_exists(VISUAL_COLLECTION_NAME):
        client.create_collection(
            collection_name=VISUAL_COLLECTION_NAME,
            vectors_config=VectorParams(size=visual_dim, distance=Distance.COSINE)
        )

    print("Loading Tencent WeMM-Embedding-4B...")
    embedder = WeMMEmbedding4BEmbedder(mrl_dim=visual_dim)

    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"Total downloaded video files: {len(video_files)}")

    ckpt_file = LOG_DIR / "indexed_videos.json"
    indexed_videos = set()
    if ckpt_file.exists():
        try:
            with open(ckpt_file, "r") as f:
                indexed_videos = set(json.load(f))
        except Exception:
            pass
    print(f"Already indexed in checkpoint: {len(indexed_videos)} videos.")

    to_process = [f for f in video_files if f.stem not in indexed_videos]
    print(f"Remaining videos to index: {len(to_process)}")

    t0_start = time.monotonic()
    total_added_points = 0

    for idx, vid_path in enumerate(to_process, start=1):
        v_id = vid_path.stem
        meta = load_video_metadata(v_id)
        title = meta.get("title", f"V3C Video {v_id}")
        desc = meta.get("description", "")
        tags = " ".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else ""
        text_blob = f"{title} {desc} {tags}".strip()

        kfs = extract_video_keyframes(vid_path, sample_interval_sec=3.0, max_frames=15)
        if not kfs:
            continue

        images = [kf["image"] for kf in kfs]
        try:
            embeddings = embedder.embed_images_batch(images)
            points: List[PointStruct] = []
            for kf, emb in zip(kfs, embeddings):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"v3c_{v_id}_{kf['frame_idx']}"))
                payload = {
                    "video_id": v_id,
                    "source_file": kf["source_file"],
                    "frame_idx": kf["frame_idx"],
                    "timestamp": kf["timestamp"],
                    "caption": title,
                    "text_blob": text_blob,
                    "modality": "visual",
                }
                points.append(PointStruct(
                    id=point_id,
                    vector=emb.tolist() if isinstance(emb, np.ndarray) else emb,
                    payload=payload,
                ))
            client.upsert(collection_name=VISUAL_COLLECTION_NAME, points=points)
            indexed_videos.add(v_id)
            total_added_points += len(points)

            if idx % 10 == 0 or idx == len(to_process):
                with open(ckpt_file, "w") as f:
                    json.dump(list(indexed_videos), f)
                print(f"[{len(indexed_videos)}/{len(video_files)}] Indexed {vid_path.name} (+{len(points)} pts | Total added: {total_added_points})", flush=True)
        except Exception as err:
            print(f"[ERROR] Failed {v_id}: {err}", flush=True)

    print(f"\n=== FULL INDEXING COMPLETED in {time.monotonic() - t0_start:.1f}s ===")


if __name__ == "__main__":
    main()

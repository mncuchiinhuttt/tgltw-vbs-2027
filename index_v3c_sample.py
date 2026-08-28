#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3C Keyframe Extraction and Multimodal Indexing for VBS 2027.
Extracts representative keyframes from downloaded V3C sample videos,
computes Qwen3-VL embeddings, and indexes them into Qdrant collection.
"""

import cv2
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
INFERENCE_DIR = REPO_ROOT / "inference-code"
DATASET_DIR = REPO_ROOT / "datasets" / "v3c"
VIDEO_DIR = DATASET_DIR / "videos"
METADATA_DIR = DATASET_DIR / "metadata"
KEYFRAME_DIR = DATASET_DIR / "keyframes"

for p in (str(REPO_ROOT), str(INFERENCE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
    VISUAL_COLLECTION_NAME, EMBEDDING_MRL_DIM,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
from models.embedding import QwenVL8BEmbedder


def extract_video_keyframes(video_path: Path, sample_interval_sec: float = 2.0, max_frames: int = 40) -> List[Dict[str, Any]]:
    """Extract keyframes from an MP4 video at fixed intervals."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0

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

        ts = round(frame_idx / fps, 2)
        save_name = f"{frame_idx}.jpg"
        save_path = out_dir / save_name

        # Save JPEG
        cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        keyframes.append({
            "video_id": video_id,
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
    """Load title and description from info JSON if available."""
    info_file = METADATA_DIR / "info" / f"{video_id}.json"
    if info_file.exists():
        try:
            with info_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main():
    print("=== V3C Keyframe Extractor & Multimodal Qdrant Indexer ===")
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY or None)

    collection_name = VISUAL_COLLECTION_NAME or "visual_keyframes_v1"
    existing_colls = [c.name for c in client.get_collections().collections]
    print(f"Active Qdrant Collections: {existing_colls}")

    if collection_name not in existing_colls:
        print(f"Creating collection '{collection_name}' (dim=2048, distance=Cosine)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
        )

    # 2. Load Embedding Model
    print("\nLoading Qwen3-VL Embedding Model...")
    embedder = QwenVL8BEmbedder(mrl_dim=EMBEDDING_MRL_DIM)

    # 3. Discover Videos
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        print(f"[WARN] No MP4 files found in {VIDEO_DIR}. Please download videos first.")
        return

    print(f"Discovered {len(video_files)} V3C video files to process.")

    total_points = 0
    t0_start = time.monotonic()

    for idx, vid_path in enumerate(video_files, start=1):
        v_id = vid_path.stem
        meta = load_video_metadata(v_id)
        title = meta.get("title", f"V3C Video {v_id}")
        desc = meta.get("description", "")
        tags = " ".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else ""
        text_blob = f"{title} {desc} {tags}".strip()

        print(f"\n[{idx}/{len(video_files)}] Processing {vid_path.name} ('{title[:40]}')...")
        kfs = extract_video_keyframes(vid_path, sample_interval_sec=2.5, max_frames=30)
        print(f"  Extracted {len(kfs)} keyframes.")

        if not kfs:
            continue

        # Compute embeddings in batches
        points: List[PointStruct] = []
        for kf in kfs:
            try:
                emb = embedder.embed_image(kf["image"])
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
                    vector=emb.tolist(),
                    payload=payload,
                ))
            except Exception as e:
                print(f"  [ERROR] Failed to embed frame {kf['frame_idx']}: {e}")

        if points:
            client.upsert(collection_name=collection_name, points=points)
            total_points += len(points)
            print(f"  Upserted {len(points)} vectors to '{collection_name}'. (Total indexed: {total_points})")

    elapsed = time.monotonic() - t0_start
    print(f"\n=== Indexing Completed in {elapsed:.1f}s ===")
    print(f"Total Keyframe Points in Qdrant: {total_points}")


if __name__ == "__main__":
    main()

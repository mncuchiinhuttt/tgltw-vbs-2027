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
from models.embedding import WeMMEmbedding4BEmbedder


def extract_video_keyframes(video_path: Path, sample_interval_sec: float = 2.0, max_frames: int = 0) -> List[Dict[str, Any]]:
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
        if max_frames > 0 and frame_count >= max_frames:
            break

    cap.release()
    return keyframes
def load_video_metadata(video_id: str) -> Dict[str, Any]:
    """Load JSON metadata if available."""
    info_path = METADATA_DIR / "info" / f"{video_id}.json"
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"title": f"V3C Video {video_id}", "description": "", "tags": []}


def main():
    print("=== V3C Multimodal Video & Keyframe Indexer ===")
    
    # 1. Initialize Qdrant Client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY or None)
    
    # Visual dimension for WeMM-Embedding-4B is 2048 (or EMBEDDING_MRL_DIM)
    visual_dim = EMBEDDING_MRL_DIM or 2048
    
    if not client.collection_exists(VISUAL_COLLECTION_NAME):
        print(f"Creating collection '{VISUAL_COLLECTION_NAME}' (dim={visual_dim})...")
        client.create_collection(
            collection_name=VISUAL_COLLECTION_NAME,
            vectors_config=VectorParams(size=visual_dim, distance=Distance.COSINE)
        )
    else:
        print(f"Collection '{VISUAL_COLLECTION_NAME}' ready.")
    
    # 2. Load Visual Embedder (Tencent WeMM-Embedding-4B)
    print("Loading Tencent WeMM-Embedding-4B embedder...")
    embedder = WeMMEmbedding4BEmbedder(mrl_dim=visual_dim)
    # 3. Discover Videos
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        print(f"[WARN] No MP4 files found in {VIDEO_DIR}. Please download videos first.")
        return

    print(f"Discovered {len(video_files)} V3C video files to process.")

    total_points = 0
    max_frames_env = max(0, int(os.getenv("V3C_MAX_FRAMES_PER_VIDEO", "0") or 0))
    t0_start = time.monotonic()

    for idx, vid_path in enumerate(video_files, start=1):
        v_id = vid_path.stem
        meta = load_video_metadata(v_id)
        title = meta.get("title", f"V3C Video {v_id}")
        desc = meta.get("description", "")
        tags = " ".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else ""
        text_blob = f"{title} {desc} {tags}".strip()

        print(f"\n[{idx}/{len(video_files)}] Processing {vid_path.name} ('{title[:40]}')...", flush=True)
        kfs = extract_video_keyframes(vid_path, sample_interval_sec=2.5, max_frames=max_frames_env)
        print(f"  Extracted {len(kfs)} keyframes.", flush=True)

        if not kfs:
            continue

        # Compute embeddings in batches
        # Compute embeddings in batches of keyframes
        images = [kf["image"] for kf in kfs]
        try:
            embeddings = embedder.embed_images_batch(images)
        except Exception as e:
            print(f"  [ERROR] Batch embedding failed: {e}")
            continue

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

        if points:
            client.upsert(collection_name=VISUAL_COLLECTION_NAME, points=points)
            total_points += len(points)
            print(f"  Upserted {len(points)} vectors to '{VISUAL_COLLECTION_NAME}'. (Total indexed: {total_points})", flush=True)
    print(f"\n=== Indexing Completed in {time.monotonic() - t0_start:.1f}s ===")
    print(f"Total Keyframe Points in Qdrant: {total_points}")


if __name__ == "__main__":
    main()

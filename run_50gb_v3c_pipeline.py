#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Tự Động Hóa VBS 2027:
1. Tải liên tục video V3C đến khi đạt mốc 50 GB.
2. Quét liên tục và trích xuất keyframes, nhúng vector WeMM-4B (2048d) và nạp vào Qdrant.
3. Ghi log tiến độ chi tiết và duy trì hoạt động bền bỉ trong nền.
"""

import os
import sys
import time
import json
import uuid
import shutil
import cv2
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import numpy as np
from PIL import Image

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

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
    VISUAL_COLLECTION_NAME, EMBEDDING_MRL_DIM,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from models.embedding import WeMMEmbedding4BEmbedder

SFTP_BASE = "sftp://ftp.itec.aau.at/V3C"
SFTP_USER = os.getenv("V3C_SFTP_USER", "v3c")
SFTP_PASS = os.getenv("V3C_SFTP_PASS", "")


def _remove_partial(local_path: Path) -> None:
    """Best-effort removal of a partial download - a locked file (e.g. AV
    scanning a just-killed curl output on Windows) must not crash the worker."""
    try:
        local_path.unlink(missing_ok=True)
    except OSError as err:
        print(f"[WARN] Could not remove partial {local_path.name}: {err}")


def run_curl_download(video_id: str) -> tuple[str, bool, int]:
    """Download single video file via curl with SFTP auth.

    On any failure the partial file is removed: a killed curl leaves a
    truncated .mp4 that the >1MB skip check would otherwise permanently
    accept as complete, index, and count toward the storage target.
    """
    local_path = VIDEO_DIR / f"{video_id}.mp4"
    if local_path.exists() and local_path.stat().st_size > 1024 * 1024:
        return video_id, True, local_path.stat().st_size

    remote_url = f"{SFTP_BASE}/small/{video_id}.mp4"
    cmd = [
        "curl", "-s", "-k",
        "--user", f"{SFTP_USER}:{SFTP_PASS}",
        "-o", str(local_path),
        "-C", "-",
        remote_url
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        ok = (res.returncode == 0 and local_path.exists() and local_path.stat().st_size > 1024 * 1024)
        if ok:
            return video_id, True, local_path.stat().st_size
        _remove_partial(local_path)
        print(f"[DOWNLOAD-FAIL] {video_id}: rc={res.returncode} {res.stderr.strip()[:200]}")
        return video_id, False, 0
    except subprocess.TimeoutExpired:
        _remove_partial(local_path)
        print(f"[DOWNLOAD-FAIL] {video_id}: timed out after 900s, partial removed")
        return video_id, False, 0
    except Exception as err:
        _remove_partial(local_path)
        print(f"[DOWNLOAD-FAIL] {video_id}: {err}")
        return video_id, False, 0


def extract_video_keyframes(video_path: Path, sample_interval_sec: float = 3.0, max_frames: int = 20) -> List[Dict[str, Any]]:
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
    if not SFTP_PASS:
        print("[ERROR] V3C_SFTP_PASS is not set - add it to .env (see .env.template) or export it.")
        print("        The official V3C SFTP credentials are published on the AAU V3C dataset page.")
        sys.exit(1)

    target_gb = 50.0
    target_bytes = int(target_gb * 1024 * 1024 * 1024)
    workers = 16

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / "pipeline_progress.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === STARTING 50GB V3C DOWNLOAD & INDEXING PIPELINE ===\n")

    # 1. Initialize Qdrant Client & Model
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY or None)

    print("Loading Tencent WeMM-Embedding-4B embedder...")
    embedder = WeMMEmbedding4BEmbedder(mrl_dim=EMBEDDING_MRL_DIM or None)

    # Probe the embedder's real output dimension instead of assuming 2048:
    # upserts against a collection with a different size fail silently,
    # per-video, forever. Also verify an existing collection matches.
    probe = embedder.embed_images_batch([np.full((32, 32, 3), 200, dtype=np.uint8)])[0]
    visual_dim = int(np.asarray(probe).shape[0])
    print(f"Embedder output dimension: {visual_dim}d")

    if client.collection_exists(VISUAL_COLLECTION_NAME):
        existing = client.get_collection(VISUAL_COLLECTION_NAME)
        existing_dim = None
        vectors_cfg = existing.config.params.vectors
        if hasattr(vectors_cfg, "size"):
            existing_dim = vectors_cfg.size
        elif isinstance(vectors_cfg, dict):
            first = next(iter(vectors_cfg.values()), None)
            existing_dim = getattr(first, "size", None)
        if existing_dim is not None and existing_dim != visual_dim:
            raise SystemExit(
                f"[ABORT] Collection '{VISUAL_COLLECTION_NAME}' is {existing_dim}d but the "
                f"embedder outputs {visual_dim}d. Point it at a fresh collection "
                f"(VISUAL_COLLECTION_NAME) or reindex - mixed-dimension upserts would all fail."
            )
    else:
        client.create_collection(
            collection_name=VISUAL_COLLECTION_NAME,
            vectors_config=VectorParams(size=visual_dim, distance=Distance.COSINE)
        )

    # 2. Candidate video discovery
    info_files = sorted((METADATA_DIR / "info").glob("*.json"))
    candidate_ids = [f.stem for f in info_files]
    print(f"Loaded {len(candidate_ids)} candidate video IDs from V3C metadata.")

    # 3. Master Pipeline Loop: Tải đến đâu index đến đó
    indexed_videos = set()
    try:
        # Load checkpoint if exists
        ckpt_file = LOG_DIR / "indexed_videos.json"
        if ckpt_file.exists():
            with open(ckpt_file, "r") as f:
                indexed_videos = set(json.load(f))
    except Exception:
        pass

    total_downloaded = sum(f.stat().st_size for f in VIDEO_DIR.glob("*.mp4"))
    print(f"Current Video Storage: {total_downloaded / (1024*1024*1024):.2f} / {target_gb:.1f} GB")

    # Disk-space preflight: fill the disk mid-download and curl writes fail,
    # partials linger, and nothing aborts cleanly.
    needed_bytes = max(target_bytes - total_downloaded, 0)
    free_bytes = shutil.disk_usage(str(VIDEO_DIR)).free
    safety_margin = 5 * 1024 * 1024 * 1024
    if needed_bytes + safety_margin > free_bytes:
        raise SystemExit(
            f"[ABORT] Not enough disk space: {free_bytes / 1024**3:.1f} GB free, "
            f"{needed_bytes / 1024**3:.1f} GB to target + {safety_margin / 1024**3:.0f} GB margin. "
            f"Free space or lower the 50 GB target before running."
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_curl_download, v_id): v_id for v_id in candidate_ids}
        
        for future in as_completed(futures):
            v_id, ok, sz = future.result()
            if ok and sz > 0:
                total_downloaded = sum(f.stat().st_size for f in VIDEO_DIR.glob("*.mp4"))
                gb = total_downloaded / (1024 * 1024 * 1024)

                free_bytes = shutil.disk_usage(str(VIDEO_DIR)).free
                if free_bytes < safety_margin:
                    print(f"\n[ABORT] Disk nearly full ({free_bytes / 1024**3:.1f} GB free); stopping downloads.")
                    for f in futures:
                        f.cancel()
                    break

                # Index this video immediately if not already indexed
                vid_path = VIDEO_DIR / f"{v_id}.mp4"
                if v_id not in indexed_videos and vid_path.exists():
                    meta = load_video_metadata(v_id)
                    title = meta.get("title", f"V3C Video {v_id}")
                    desc = meta.get("description", "")
                    tags = " ".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else ""
                    text_blob = f"{title} {desc} {tags}".strip()

                    kfs = extract_video_keyframes(vid_path, sample_interval_sec=3.0, max_frames=20)
                    if kfs:
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
                        except Exception as err:
                            print(f"[ERROR] Indexing failed for {v_id}: {err}")

                    # Save checkpoint
                    with open(LOG_DIR / "indexed_videos.json", "w") as f:
                        json.dump(list(indexed_videos), f)

                msg = f"[{len(indexed_videos)} Indexed] Downloaded {v_id}.mp4 | Total Storage: {gb:.2f} / {target_gb:.1f} GB\n"
                print(msg.strip(), flush=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(msg)

            if total_downloaded >= target_bytes:
                print(f"\n[SUCCESS] Reached target size {target_gb:.1f} GB.")
                for f in futures:
                    f.cancel()
                break

    print(f"\n=== PIPELINE FINISHED: Indexed {len(indexed_videos)} videos into Qdrant ===")


if __name__ == "__main__":
    main()

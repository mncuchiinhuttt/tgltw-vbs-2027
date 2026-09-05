#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3C Dataset Sample Downloader for VBS 2027 Experiments.
Downloads official V3C videos and metadata from ftp.itec.aau.at with multithreaded curl.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "datasets" / "v3c"
VIDEO_DIR = DATASET_DIR / "videos"
METADATA_DIR = DATASET_DIR / "metadata"

SFTP_BASE = "sftp://ftp.itec.aau.at/V3C"
SFTP_USER = os.getenv("V3C_SFTP_USER", "v3c")
SFTP_PASS = os.getenv("V3C_SFTP_PASS", "")


def run_curl_download(video_id: str) -> tuple[str, bool, int]:
    """Download single video file via curl with SFTP auth."""
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
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        ok = (res.returncode == 0 and local_path.exists() and local_path.stat().st_size > 1024 * 1024)
        sz = local_path.stat().st_size if ok else 0
        return video_id, ok, sz
    except Exception:
        return video_id, False, 0


def main():
    target_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    target_bytes = int(target_gb * 1024 * 1024 * 1024)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    if not SFTP_PASS:
        print("[ERROR] V3C_SFTP_PASS is not set - add it to .env (see .env.template) or export it.")
        print("        The official V3C SFTP credentials are published on the AAU V3C dataset page.")
        sys.exit(1)

    print(f"=== V3C Multithreaded Dataset Downloader ===")
    print(f"Target Size: {target_gb:.1f} GB | Workers: {workers}")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect candidate video IDs from extracted metadata
    info_files = sorted((METADATA_DIR / "info").glob("*.json"))
    if not info_files:
        print("[ERROR] No info json files found. Please extract info.tar.gz first.")
        sys.exit(1)

    candidate_ids = [f.stem for f in info_files]
    print(f"Loaded {len(candidate_ids)} candidate video IDs from V3C metadata.")
    current_size = sum(f.stat().st_size for f in VIDEO_DIR.glob("*.mp4"))
    print(f"Existing video storage: {current_size / (1024*1024*1024):.2f} GB")

    if current_size >= target_bytes:
        print("Target storage already satisfied.")
        return

    # 3. Concurrent Download Loop
    total_downloaded = current_size
    start_t = time.monotonic()
    completed_videos = 0

    print("\nStarting concurrent downloads...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_curl_download, v_id): v_id for v_id in candidate_ids}
        for future in as_completed(futures):
            v_id, ok, sz = future.result()
            if ok and sz > 0:
                completed_videos += 1
                total_downloaded += sz
                mb = sz / (1024 * 1024)
                gb = total_downloaded / (1024 * 1024 * 1024)
                print(f"[{completed_videos}] Downloaded {v_id}.mp4 ({mb:.1f} MB) | Total: {gb:.2f} / {target_gb:.1f} GB")

            if total_downloaded >= target_bytes:
                print(f"\n[SUCCESS] Reached target size {target_gb:.1f} GB ({total_downloaded / (1024*1024*1024):.2f} GB).")
                for f in futures:
                    f.cancel()
                break

    elapsed = time.monotonic() - start_t
    stored_count = len(list(VIDEO_DIR.glob("*.mp4")))
    total_gb = sum(f.stat().st_size for f in VIDEO_DIR.glob("*.mp4")) / (1024 * 1024 * 1024)
    print(f"\n=== Download Finished in {elapsed:.1f}s ===")
    print(f"Total V3C Videos stored: {stored_count}")
    print(f"Total Video Storage: {total_gb:.2f} GB")


if __name__ == "__main__":
    main()

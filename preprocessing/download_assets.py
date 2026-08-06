#!/usr/bin/env python3
"""Download optional preprocessing model assets.

The large files are intentionally not committed to git.  The downloader is
idempotent and writes to the path consumed by ``TRANSNETV2_MODEL_PATH``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests


TRANSNETV2_REVISION = "a97542e4eb22e3af904ac13b10cf06da507e2ff1"
TRANSNETV2_SHA256 = "46520d66d4bf60414a4d82e0e94a92442ff950e34517a3718b2e54815e642b53"
TRANSNETV2_URL = (
    "https://huggingface.co/MiaoshouAI/transnetv2-pytorch-weights/resolve/"
    f"{TRANSNETV2_REVISION}/transnetv2-pytorch-weights.pth?download=true"
)
DEFAULT_TRANSNETV2_PATH = Path(__file__).resolve().parent.parent / "weights" / "transnetv2-pytorch-weights.pth"


def download(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}")
    with requests.get(url, stream=True, timeout=(20, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        received = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r  {received / total * 100:5.1f}%", end="", flush=True)
    print()
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != TRANSNETV2_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {destination}: expected {TRANSNETV2_SHA256}, got {digest}"
        )
    temporary.replace(destination)
    print(f"Saved {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MiB, sha256={digest[:16]}...).")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download optional VBS preprocessing model assets")
    parser.add_argument("--transnetv2", action="store_true", help="Download the TransNetV2 PyTorch checkpoint")
    parser.add_argument("--output", type=Path, default=DEFAULT_TRANSNETV2_PATH)
    args = parser.parse_args()
    if not args.transnetv2:
        parser.error("choose --transnetv2")
    if args.output.is_file() and args.output.stat().st_size > 0:
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
        if digest != TRANSNETV2_SHA256:
            raise RuntimeError(
                f"Existing TransNetV2 checkpoint has unexpected sha256: {digest}"
            )
        print(f"TransNetV2 checkpoint already exists and passed checksum: {args.output}")
        return
    download(TRANSNETV2_URL, args.output)


if __name__ == "__main__":
    main()

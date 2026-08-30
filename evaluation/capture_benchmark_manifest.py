#!/usr/bin/env python3
"""Capture hashes and runtime metadata for an AEGIS offline benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATHS = (
    "queries/vbs_rag_benchmark.json",
    "evaluation/eval_queries_real_v3c.json",
    "inference-code/config.py",
    "inference-code/search/hybrid_search.py",
    "inference-code/search/reranker.py",
    "evaluation/run_rag_benchmark.py",
    "evaluation/run_comprehensive_ablation.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture(root: Path, paths: tuple[str, ...]) -> dict:
    files = {}
    for relative in paths:
        path = root / relative
        if path.is_file():
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        else:
            files[relative] = {"missing": True}
    return {
        "schema_version": "aegis-benchmark-manifest-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(root),
        "python": sys.version,
        "platform": platform.platform(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(capture(root, tuple(args.paths)), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

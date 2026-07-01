#!/usr/bin/env python3
import importlib.util
import os
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT_DIR / "weights"
DATASETS_DIR = ROOT_DIR / "datasets"
DEPS_DIR = ROOT_DIR / ".download_assets_deps"

if DEPS_DIR.exists() and str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

def ensure_package(package_name: str) -> None:
    if importlib.util.find_spec(package_name) is not None:
        return

    print(f"Installing {package_name}...")
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            str(DEPS_DIR),
            package_name,
        ]
    )

    if str(DEPS_DIR) not in sys.path:
        sys.path.insert(0, str(DEPS_DIR))

    importlib.invalidate_caches()

    if importlib.util.find_spec(package_name) is None:
        raise ModuleNotFoundError(
            f"Unable to import {package_name} after installing it into {DEPS_DIR}"
        )

def download_model(repo_id: str, local_name: str):
    ensure_package("huggingface_hub")
    from huggingface_hub import snapshot_download
    
    dest_path = WEIGHTS_DIR / local_name
    print(f"\n--- Downloading {repo_id} to {dest_path} ---")
    dest_path.mkdir(parents=True, exist_ok=True)
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest_path),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"Download complete: {local_name}")

def main():
    print("=== AIC2026 Assets Downloader ===")
    
    # Create directories
    WEIGHTS_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    
    # Download weights
    # 1. PhoWhisper
    download_model("vinai/PhoWhisper-large", "PhoWhisper-large")
    
    # 2. CLAP Environmental model
    download_model("laion/clap-htsat-fused", "clap-htsat-fused")
    
    # 3. Rex-Omni Zero-shot Detection model
    download_model("IDEA-Research/Rex-Omni", "Rex-Omni")
        
    print("\n=== All requested models checked/downloaded! ===")
    print(f"Global weights stored in: {WEIGHTS_DIR}")
    print(f"Place your raw dataset files in: {DATASETS_DIR}")

if __name__ == "__main__":
    main()

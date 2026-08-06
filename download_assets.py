#!/usr/bin/env python3
import importlib.util
import os
import sys
import subprocess
import shutil
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

def get_hf_executable() -> str:
    exe = shutil.which("hf")
    if exe:
        return exe
    local_path = os.path.expanduser("~/.local/bin/hf")
    if os.path.exists(local_path):
        return local_path
    return "hf"

def download_model(repo_id: str, local_name: str, token: str = None):
    dest_path = WEIGHTS_DIR / local_name
    print(f"\n--- Downloading model {repo_id} to {dest_path} using hf CLI ---")
    dest_path.mkdir(parents=True, exist_ok=True)
    
    hf_exe = get_hf_executable()
    cmd = [
        hf_exe, "download", repo_id,
        "--local-dir", str(dest_path),
        "--type", "model"
    ]
    
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
        
    subprocess.check_call(cmd, env=env)
    print(f"Download complete: {local_name}")

def download_and_unzip_m2d_clap():
    url = "https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
    dest_dir = WEIGHTS_DIR / "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
    zip_path = WEIGHTS_DIR / "m2d_clap_vit_base.zip"
    
    if (dest_dir / "checkpoint-30.pth").exists():
        print(f"\n--- M2D-CLAP checkpoint already exists in {dest_dir} ---")
        return
        
    print(f"\n--- Downloading M2D-CLAP model from {url} ---")
    ensure_package("requests")
    import requests
    import zipfile
    import shutil as sh
    
    # Download zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                
    print(f"Extracting M2D-CLAP model...")
    # Extract to a temp directory first
    temp_extract_dir = WEIGHTS_DIR / "_temp_m2d_clap"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    # Find checkpoint-30.pth inside temp_extract_dir
    found_ckpt = None
    for p in temp_extract_dir.glob("**/checkpoint-30.pth"):
        found_ckpt = p
        break
        
    if found_ckpt:
        # Move it to weights/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth
        dest_dir.mkdir(parents=True, exist_ok=True)
        sh.move(str(found_ckpt), str(dest_dir / "checkpoint-30.pth"))
        print(f"Successfully placed checkpoint-30.pth in {dest_dir}")
    else:
        print("Warning: checkpoint-30.pth not found in zip archive.")
        
    # Clean up temp dir and zip file
    if temp_extract_dir.exists():
        sh.rmtree(temp_extract_dir)
    if zip_path.exists():
        zip_path.unlink()
        
    print("M2D-CLAP download and extraction complete.")

def download_real_esrgan(local_name: str):
    dest_path = WEIGHTS_DIR / local_name
    if dest_path.exists():
        print(f"\n--- Real-ESRGAN weights already exist at {dest_path} ---")
        return

    url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{local_name}"
    print(f"\n--- Downloading Real-ESRGAN weights from {url} ---")
    ensure_package("requests")
    import requests

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Real-ESRGAN weights downloaded to {dest_path}")

def ensure_ffmpeg():
    # Check if ffmpeg is available in PATH
    if shutil.which("ffmpeg") is not None:
        return
        
    bin_dir = ROOT_DIR / "bin"
    ffmpeg_path = bin_dir / "ffmpeg"
    
    if ffmpeg_path.exists():
        return
        
    print("\n--- ffmpeg not found on system. Downloading macOS static binary ---")
    bin_dir.mkdir(exist_ok=True)
    zip_path = bin_dir / "ffmpeg.zip"
    
    url = "https://evermeet.cx/ffmpeg/getrelease/zip"
    ensure_package("requests")
    import requests
    import zipfile
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print("Extracting ffmpeg binary...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(bin_dir)
            
        # Make it executable
        os.chmod(ffmpeg_path, 0o755)
        
        # Clean up
        if zip_path.exists():
            zip_path.unlink()
            
        # Quarantine workaround for macOS Gatekeeper
        print("Removing quarantine attribute on macOS...")
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(ffmpeg_path)], capture_output=True)
        
        print("ffmpeg static binary successfully installed at:", ffmpeg_path)
    except Exception as e:
        print(f"Warning: Failed to download static ffmpeg: {e}. You may need to install it manually.")

def download_dataset(repo_id: str, local_name: str, token: str = None):
    dest_path = DATASETS_DIR / local_name
    print(f"\n--- Downloading dataset {repo_id} to {dest_path} using hf CLI ---")
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Download metadata and everything EXCEPT videos first
    hf_exe = get_hf_executable()
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
        
    cmd_exclude = [
        hf_exe, "download", repo_id,
        "--local-dir", str(dest_path),
        "--type", "dataset",
        "--exclude", "video/*"
    ]
    subprocess.check_call(cmd_exclude, env=env)
    
    # 2. Download only the first 2000 videos (video0.mp4 to video1999.mp4)
    print("Downloading first 2000 videos in batches...")
    batch_size = 100
    for i in range(0, 2000, batch_size):
        videos_batch = [f"video/video{j}.mp4" for j in range(i, min(i + batch_size, 2000))]
        cmd_videos = [
            hf_exe, "download", repo_id,
        ] + videos_batch + [
            "--local-dir", str(dest_path),
            "--type", "dataset"
        ]
        subprocess.check_call(cmd_videos, env=env)
        
    # 3. Clean up any extra videos (>= 2000) to keep the limit strictly at 2000
    video_dir = dest_path / "video"
    if video_dir.exists():
        print("Cleaning up any extra videos beyond the 2000 limit...")
        for p in video_dir.glob("video*.mp4"):
            try:
                num = int(p.stem.replace("video", ""))
                if num >= 2000:
                    p.unlink()
            except ValueError:
                pass
                
    print("Dataset download complete: msrvtt-corpus (limited to 2000 videos)")

def load_env_values():
    env_vars = {}
    
    # Load from root .env if exists
    root_env = ROOT_DIR / ".env"
    if root_env.exists():
        with open(root_env, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
                    
    # Load from preprocessing/.env if exists (overriding defaults)
    pre_env = ROOT_DIR / "preprocessing" / ".env"
    if pre_env.exists():
        with open(pre_env, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
                    
    return env_vars

def main():
    print("=== AIC2026 Assets Downloader ===")
    
    # Create directories
    WEIGHTS_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    
    # Load model configuration from .env files
    env_vars = load_env_values()
    qwen_vlm_id = env_vars.get("QWEN_VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Thinking")
    qwen_embed_id = env_vars.get("QWEN_EMBEDDING_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B")
    asr_model_id = env_vars.get("ASR_MODEL_ID", "deepdml/faster-whisper-large-v3-turbo-ct2")
    yoloe_id = env_vars.get("YOLOE_MODEL_ID", "yoloe-26x-seg.pt")
    sam3_id = env_vars.get("SAM3_MODEL_ID", "facebook/sam3")
    fallback_vlm_id = env_vars.get("FALLBACK_VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    real_esrgan_id = env_vars.get("REAL_ESRGAN_MODEL_ID", "RealESRGAN_x4plus.pth")
    vlm_option = env_vars.get("VLM_OPTION", "openai")
    embedding_option = env_vars.get("EMBEDDING_OPTION", "local")
    hf_token = env_vars.get("HF_TOKEN")

    # Download weights
    # 1. Whisper large-v3-turbo (CTranslate2 format, for faster-whisper)
    download_model(asr_model_id, asr_model_id.split("/")[-1], token=hf_token)

    # 2. M2D-CLAP Environmental model (unzipped from URL)
    download_and_unzip_m2d_clap()

    # 3. YOLOE-26 zero-shot detector - fetched via the ultralytics asset
    # registry (GitHub releases), not the Hugging Face Hub like the other
    # models here, so it's downloaded directly to its expected weights/ path
    ensure_package("ultralytics")
    from ultralytics import YOLOE
    yoloe_path = WEIGHTS_DIR / yoloe_id
    if not yoloe_path.exists():
        print(f"\n--- Downloading YOLOE detector {yoloe_id} to {yoloe_path} ---")
        YOLOE(str(yoloe_path))
    else:
        print(f"\n--- YOLOE detector already exists at {yoloe_path} ---")

    # 4. Qwen VLM (if VLM_OPTION=local)
    if vlm_option == "local":
        download_model(qwen_vlm_id, qwen_vlm_id.split("/")[-1], token=hf_token)
    else:
        print(f"\nSkipping local Qwen VLM download (VLM_OPTION={vlm_option}).")
        print("To download the local VLM, set VLM_OPTION=local in preprocessing/.env")
        
    # 5. Qwen Embedding (if EMBEDDING_OPTION=local)
    if embedding_option == "local":
        download_model(qwen_embed_id, qwen_embed_id.split("/")[-1], token=hf_token)
    else:
        print(f"\nSkipping local Qwen embedding download (EMBEDDING_OPTION={embedding_option}).")

    # 6. SAM3 region-proposal pre-filter (gated - accept the license at
    # https://huggingface.co/facebook/sam3 and set HF_TOKEN before this runs)
    if not hf_token:
        print("\nWarning: HF_TOKEN not set - SAM3 (facebook/sam3) is a gated repo and this "
              "download will likely fail until you accept its license and set HF_TOKEN.")
    download_model(sam3_id, "sam3", token=hf_token)

    # 7. Fallback VLM for low-confidence OCR crops (SmolVLM2 by default)
    download_model(fallback_vlm_id, fallback_vlm_id.split("/")[-1], token=hf_token)

    # 8. Real-ESRGAN x4 weights - conditional Super-Resolution for small OCR crops
    download_real_esrgan(real_esrgan_id)

    # Ensure ffmpeg is installed
    ensure_ffmpeg()
    
    # Download dataset
    download_dataset("Tevatron/msrvtt-corpus", "msrvtt-corpus", token=hf_token)
        
    print("\n=== All requested models and datasets checked/downloaded! ===")
    print(f"Global weights stored in: {WEIGHTS_DIR}")
    print(f"Dataset stored in: {DATASETS_DIR / 'msrvtt-corpus'}")

if __name__ == "__main__":
    main()

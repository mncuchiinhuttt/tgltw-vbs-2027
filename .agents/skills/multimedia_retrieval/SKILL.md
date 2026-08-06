---
name: Multimedia Retrieval Agent
description: Guidelines and specialized instructions for developing, testing, and troubleshooting the HCMC AI Challenge 2026 Multimedia Retrieval codebase (preprocessing, Qdrant indexing, and query inference).
---

# Multimedia Retrieval Agent Skill

This skill helps agents understand the structure and execution of the Multimedia Retrieval pipeline inside the workspace.

## System Architecture

The project consists of three main parts:
1. **Shared models (`models/`)**: Unified python model loaders for Qwen3-VL, OpenAI-compatible VLM (OpenAI or alternative providers via `OPENAI_BASE_URL`), ASR (faster-whisper running Whisper large-v3-turbo), sound embeddings (CLAP), and zero-shot object detection (YOLOE-26).
2. **Preprocessing (`preprocessing/`)**: Extracts keyframes, descriptors, OCR, and sound embeddings from raw videos/audio, indexing them into Qdrant.
3. **Inference (`inference-code/`)**: Processes Textual-KIS (Type 1), VQA (Type 2), and Temporal-Alignment (Type 3) search queries.

## Key Run Instructions

### 1. Model & Data Download
Use the central downloader to pull checkpoints:
```bash
python download_assets.py
```
*Weights are stored in the ignored `/weights/` directory.*

### 2. Startup Database (Qdrant)
Run the helper script inside `preprocessing` to start the vector DB locally:
```bash
cd preprocessing
./host_qdrant.sh
```
*Connects on `localhost:6333` and maps persistent data storage to `./qdrant_storage`.*

### 3. Running Preprocessing Pipeline
Configure `preprocessing/.env` and execute:
```bash
cd preprocessing
source venv/bin/activate
python main.py --data_dir ../datasets
```

### 4. Executing Search Queries (Inference)
Configure `inference-code/.env` and run CLI queries:
```bash
cd inference-code
# Type 1
python main.py --type 1 --query "search query text"
# Type 2
python main.py --type 2 --query "VQA question text" --dataset_dir ../datasets
# Type 3
python main.py --type 3 --query "chronological event descriptions"
```

## Developing/Editing Guidelines

- **Shared models**: Do not duplicate model wrappers. All VLM, ASR, CLAP, and Object Detection logic must reside inside the root `/models/` folder.
- **Dynamic sys.path**: When writing scripts inside subdirectories, always append the parent directory path to `sys.path` to enable clean `import models` statements:
  ```python
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  ```
- **OCR Text Processing**: OCR output must be normalized to Unicode **NFC** form while preserving the recognized text. Do not apply language-specific accent stripping or transliteration before indexing.
- **VQA Cropping**: For Type 2 queries, always check if sub-query objects are detected by `ObjectDetector` (YOLOE-26). If so, crop the bounding box region (`xmin, ymin, xmax, ymax`) and pass the cropped image to the VLM rather than the full image.

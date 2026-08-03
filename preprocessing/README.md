# Preprocessing & Indexing Pipeline (HCMC AI Challenge 2026)

This module handles the extraction, description, embedding generation, and Qdrant indexing of video, image, and audio files.

## Project Structure

```
preprocessing/
├── config.py              # Configuration settings and model selection switches
├── main.py                # Main orchestrator script to run the preprocessing pipeline
├── requirements.txt       # Python dependencies
├── setup.sh               # Environment configuration script
├── CHANGELOG.md           # Log of project changes
├── host_qdrant.sh         # Starts Qdrant (via Docker or standalone binary download)
├── docker-compose.yml     # Docker Compose configuration for hosting Qdrant
├── video/
│   ├── scene_detector.py  # Scene boundary detection and adaptive keyframe sampling
│   ├── ocr.py             # OCR text extraction and Vietnamese normalizations
│   └── captioner.py       # Temporal/Scene narrative captions & structured attributes
├── audio/
│   └── audio_processor.py # Audio transcription and CLAP ambient features pipeline
└── indexing/
    └── indexer.py         # Qdrant vector database client connection and indexer
```

*(Note: Shared models logic has been moved to the root `/models/` directory, and `host_vllm.sh` lives at the repo root since it's shared with `inference-code/` too).*

## Features

1. **Scene Boundary Detection & Adaptive Keyframe Sampling**: Cuts video using `PySceneDetect`, then for each scene estimates visual variance with a lightweight CLIP pass to size a per-scene keyframe budget (1 frame for static scenes, up to `KEYFRAME_MAX_BUDGET`=8 for dynamic ones), and selects that many via farthest-point sampling in the Qwen3-Embedding-VL-8B space.
2. **Flexible VLM, Embedding & Object Detection Engines**:
   - VLM options (`VLM_OPTION`): local offline HuggingFace models (`generate_batch()` runs one true batched `model.generate()` call) or any OpenAI-compatible API (`OPENAI_BASE_URL`/`OPENAI_VLM_MODEL_NAME` - OpenAI itself, an alternative provider such as QwenCloud, or a self-hosted vLLM server for batch inference via the root `host_vllm.sh`). `generate_batch()` issues concurrent requests (`VLM_BATCH_CONCURRENCY`) so a batch-serving backend gets real throughput benefit.
   - Embedding options (`EMBEDDING_OPTION`): local `QwenVL8BEmbedder` or `DashScopeCloudEmbedder` (cloud, model configurable via `DASHSCOPE_EMBEDDING_MODEL_NAME`, no local weights - useful to cut memory pressure when running several large local models at once).
   - Object Detection: YOLOE-26 (open-vocabulary, text-prompted, NMS-free) to locate objects zero-shot based on label lists, with a supplementary tiled detection pass for small objects (e.g. license plates) and optional example-crop visual prompting for categories that are awkward to phrase in text.
3. **OCR via PP-OCRv6**: Detection + recognition run directly through PP-OCRv6 instead of the VLM; only crops recognized below `OCR_REC_SCORE_THRESHOLD` get escalated to the VLM for a re-read. Custom Vietnamese normalizations (Unicode NFC) index both accented and unaccented terms for robust BM25 search. Optional overlapping-tile pass (`OCR_USE_TILING`, off by default) for small/corner text.
4. **Unified Per-Frame VLM Analysis**: One JSON call per keyframe (caption + objects/colors/count/scene_type/attributes) instead of two separate calls, batched across a scene's keyframes via `generate_batch()`.
5. **Speech & Audio Feature Extractors**: Speech transcription via faster-whisper (Whisper large-v3-turbo, with VAD + confidence filtering), environment audio indexing via M2D-CLAP.
6. **Qdrant Vector Database Integration**: Creates unified `visual_index` and `audio_env_index` collections and loads detailed metadata payload alongside vectors.

## Installation

Run the setup script to initialize the virtual environment and install all packages:

```bash
chmod +x setup.sh
./setup.sh
```

Activate the environment:

```bash
source venv/bin/activate
```

## Self-hosting Qdrant

You can host Qdrant locally/on your server using Docker or as a standalone binary:

Run the script to launch Qdrant:

```bash
chmod +x host_qdrant.sh
./host_qdrant.sh
```

* **Docker Mode**: If Docker or Docker Compose is installed, Qdrant will start in a container with persistent storage mapped to `./qdrant_storage`.
* **Standalone Binary Mode**: If Docker is not found, the script automatically downloads the correct precompiled binary from Qdrant's GitHub Releases (matched to your OS & architecture), extracts it into `./qdrant_bin`, and starts it as a background process (`nohup`).

Once started, access the Web Dashboard at: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## Self-hosting the VLM via vLLM (optional, GPU required)

`host_vllm.sh` lives at the repo root (shared with `inference-code/`) rather than here - see the root `README.md` for setup instructions.

## Usage

1. Configure your APIs, select your model backends (Local vs API), and adjust thresholds in `config.py`.
2. Run the main processing script pointing to your dataset directory:

```bash
python main.py --data_dir /path/to/raw/dataset --temp_dir ./temp
```

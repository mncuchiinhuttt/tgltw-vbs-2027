# Preprocessing & Indexing Pipeline (VBS 2027)

This module handles the extraction, description, embedding generation, and Qdrant indexing of video, image, and audio files.

## Project Structure

```
preprocessing/
├── config.py              # Configuration settings and model selection switches
├── main.py                # Main orchestrator script to run the preprocessing pipeline
├── download_assets.py     # Download optional TransNetV2 checkpoint
├── v3c_assets.py          # Optional V3C shot/keyframe/metadata/ASR asset adapter
├── requirements.txt       # Python dependencies
├── setup.sh               # Environment configuration script
├── CHANGELOG.md           # Log of project changes
├── host_qdrant.sh         # Starts Qdrant (via Docker or standalone binary download)
├── docker-compose.yml     # Docker Compose configuration for hosting Qdrant
├── video/
│   ├── scene_detector.py  # TransNetV2/PySceneDetect shot detection and keyframe sampling
│   ├── transnet_detector.py # Streaming TransNetV2 adapter with fallback
│   ├── ocr.py             # OCR text extraction and Unicode normalization
│   └── captioner.py       # Temporal/Scene narrative captions & structured attributes
├── audio/
│   └── audio_processor.py # Audio transcription and CLAP ambient features pipeline
└── indexing/
    ├── indexer.py         # Qdrant vector database client connection and indexer
    └── heagle.py          # H-EAGLE-lite shot aggregation helpers
```

*(Note: Shared models logic has been moved to the root `/models/` directory, and `host_vllm.sh` lives at the repo root since it's shared with `inference-code/` too).*

## Features

1. **Shot Boundary Detection & Adaptive Keyframe Sampling**: Uses official V3C shot boundaries and representative keyframes when they are mounted. For raw videos it uses streaming TransNetV2 by default, with PySceneDetect as a runtime fallback. Local candidates use CLIP variance to size a per-shot budget and Tencent WeMM-Embedding-4B farthest-point sampling, with a small configurable Laplacian-sharpness bonus (`KEYFRAME_SHARPNESS_WEIGHT`) to prefer readable frames.
2. **Flexible VLM, WeMM Embedding & Object Detection Engines**:
   - VLM options (`VLM_OPTION`): local offline HuggingFace models or any OpenAI-compatible API.
   - The visual/text embedding is fixed to Tencent WeMM-Embedding-4B; Qwen is never used for embeddings.
   - Object Detection: YOLOE-26 (open-vocabulary, text-prompted, NMS-free) with optional tiled detection.
3. **OCR via PP-OCRv6**: Detection + recognition run directly through PP-OCRv6; only low-confidence crops below `OCR_REC_SCORE_THRESHOLD` get escalated to the lightweight fallback VLM for a re-read. OCR text is preserved after generic Unicode NFC normalization without language-specific accent mapping. Optional overlapping-tile pass (`OCR_USE_TILING`, off by default) handles small/corner text.
4. **Unified Per-Frame VLM Analysis**: One JSON call per keyframe (caption + objects/colors/count/scene_type/attributes) instead of two separate calls, batched across a scene's keyframes via `generate_batch()`.
5. **Speech & Audio Feature Extractors**: Speech transcription via faster-whisper (Whisper large-v3-turbo, with VAD + confidence filtering), environment audio indexing via M2D-CLAP.
6. **Qdrant Vector Database Integration**: Creates unified `visual_index`, `audio_env_index`, and separate `vbs_shot_index` collections. Frame payloads keep a stable `shot_id`; H-EAGLE-lite stores normalized shot aggregates and links back to representative frame point IDs. All points upload in configurable batches (`QDRANT_UPSERT_BATCH_SIZE`) with a flush at each video/process boundary.

## Optional V3C official assets

V3C/VBS distributions can be mounted beside the raw videos with these
directories:

```text
assets/
├── msb/<video-stem>.txt       # shot rows: shot id, start, end
├── keyframes/<video-stem>/*   # one representative image per shot
├── metadata/<video-stem>.json  # title/description/category metadata
└── asr/<video-stem>.csv        # start,end,transcript
```

Set `V3C_ASSETS_DIR=/path/to/assets` in `preprocessing/.env`, or leave it
blank to probe the `--data_dir` itself. `V3C_ASSETS_ENABLED=true` enables the
adapter. Each asset family is independent: malformed/missing shot files use
TransNetV2 and then PySceneDetect, missing ASR uses local faster-whisper, and an ambiguous
keyframe mapping uses raw-video extraction. Official keyframe timestamps are
the shot midpoint unless the dataset provides a more precise mapping, so the
payload also stores `shot_id` and `asset_source` for auditability.

### TransNetV2 and H-EAGLE-lite

Download the optional PyTorch checkpoint once before processing raw videos:

```bash
python preprocessing/download_assets.py --transnetv2
```

The detector reads low-resolution frames in a bounded streaming window, so it
does not load an entire long video into RAM. Configure `SHOT_DETECTOR`,
`TRANSNETV2_MODEL_PATH`, `TRANSNETV2_DEVICE`, and `TRANSNETV2_THRESHOLD` in
`.env`. Official V3C `msb` files still take precedence, and setting
`SHOT_DETECTOR=pyscenedetect` is the immediate rollback.

Qdrant schema mismatches fail safely by default. Set
`QDRANT_ALLOW_RECREATE=true` only when intentionally rebuilding an entire
collection. For a deliberate one-video rebuild after changing detector
weights/thresholds, set `QDRANT_REBUILD_VIDEO_ON_START=true`; it removes that
video's visual and shot points before processing it.

H-EAGLE-lite is preprocessing-only: each selected frame belongs to a shot
and a normalized aggregate is written to `vbs_shot_index`. Query-time
coarse-to-fine routing is implemented but off by default; enable
`HEAGLE_LITE_ENABLED=true` in `inference-code/.env` only after measuring
recall and latency on the real corpus. The full H-EAGLE narrative-action/VLM
level is intentionally deferred.

Learned saliency is intentionally not part of the default pipeline. DAKE,
Laplacian sharpness and embedding-space diversity remain the low-cost,
text-preserving keyframe selector; a saliency model must pass a separate
text/location recall benchmark before it can be enabled.

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

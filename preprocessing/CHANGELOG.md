# Changelog

All notable changes to this project will be documented in this file.

## [1.3.0] - 2026-07-09

### Changed
- Swapped zero-shot object detector from NVIDIA LocateAnything-3B to **YOLOE-26** (`yoloe-26x-seg.pt`) - open-vocabulary, text-prompted, NMS-free end-to-end, runs natively on CUDA/MPS/CPU with no custom kernel requirements (LocateAnything-3B required NVIDIA's `magi_attention`/flash-attn kernels, unavailable outside Hopper/Blackwell GPUs, and needed extensive `trust_remote_code` compatibility patching against current `transformers`).
- Object detection prompts now embed English translations (`OBJECT_DETECTION_PROMPTS_EN`) for better CLIP text-encoder alignment, while the reported/indexed labels stay Vietnamese for BM25 matching against Vietnamese queries.
- Real per-detection confidence scores (`DETECTION_CONF_THRESHOLD`) now flow through detection payloads, replacing the previous hardcoded `conf=0.90` stub.
- Fixed `QwenVL8BEmbedder` to use the model's own documented last-token-pooling API instead of a nonexistent `get_text_features`/mismatched `get_image_features` approach; both text and image embeddings now correctly share a single 4096-dim space (previously assumed, and partially indexed at, 1536d).
- Fixed `M2DClapEmbedder` to pass `flat_features=True` to `PortableM2D` - without it, the CLAP head (`audio_proj`/`sem_token`) crashed on every single audio embedding call with a tensor-shape mismatch (768 vs 3840), regardless of segment duration. Also corrected the assumed embedding dimension from 512 to the model's actual 768.
- `QdrantIndexer` now dynamically probes and auto-recreates both `visual_index` and `audio_env_index` on dimension mismatch, instead of hardcoding sizes that silently drifted out of sync with the actual embedding models.
- Added `EMBEDDING_OPTION` (`"local"`/`"cloud"`) and `DashScopeCloudEmbedder` - a temporary drop-in for `QwenVL8BEmbedder` using QwenCloud's `tongyi-embedding-vision-plus` multimodal embedding API, for reducing local memory footprint when several large local models are loaded at once.
- `OpenAIVLM` now supports a configurable `base_url`/model name (`OPENAI_BASE_URL`, `OPENAI_VLM_MODEL_NAME`), enabling any OpenAI-compatible provider (e.g. QwenCloud) instead of only OpenAI directly.

### Added
- `ObjectDetector.detect_tiled()` - supplementary overlapping-tile detection pass to improve recall for small objects (e.g. license plates) that get lost when a full frame is downscaled to the detector's input size.
- `ObjectDetector.detect_visual_prompt()` - detect via example image crops instead of a text description, for categories that are awkward to phrase in words.
- `TILED_DETECTION_LABELS`/`TILED_DETECTION_LABELS_EN` config, wired into a new `detect_objects()` helper in `main.py` that merges full-frame and tiled detections with per-label IoU dedup.

### Fixed
- Added missing `DETECTOR_OPTION` config value - referenced by `ObjectDetector`/`main.py` but never defined, causing an `ImportError` on startup.
- `models/asr.py`: PhoWhisper's ASR pipeline shells out to a bare `ffmpeg` command internally with no way to point it at a specific binary; now falls back to the workspace's bundled `bin/ffmpeg` (prepended to `PATH`) when system `ffmpeg` is absent.
- ASR crash when Whisper can't predict an end timestamp for a chunk (returns `None`) - now falls back to the segment's start time instead of propagating `None` into downstream timestamp arithmetic.

## [1.2.0] - 2026-07-05

### Changed
- Swapped zero-shot object detector from Rex-Omni to **NVIDIA LocateAnything-3B** (`nvidia/LocateAnything-3B`) across the preprocessing config, download pipeline, and model wrappers.
- Configured LocateAnything model loading via `transformers.pipeline` and manual `AutoModel` fallback.

## [1.1.0] - 2026-07-03

### Changed
- Aligned image preprocessing flow with the latest Notion guidelines.
- Simplified payload structure by merging `source_id` into the existing `source_file` field.
- Updated image `text_blob` creation to format text using whitespace separators without temporal elements.

## [1.0.0] - 2026-06-30

### Added
- Created complete preprocessing pipeline structures.
- Added scene boundary detection using `PySceneDetect`.
- Integrated Cosine Similarity based keyframe diversity selection.
- Created multi-backend Vision-Language Models (VLM) supporting local Qwen3-VL and OpenAI GPT 5.5 Pro API.
- Integrated Rex-Omni (offline zero-shot open source) object detection.
- Developed OCR extraction and Vietnamese Unicode NFC normalization (including unaccented text indexing for BM25).
- Created PhoWhisper speech transcription and CLAP environmental audio embedding extraction.
- Implemented indexing functionality pushing payloads and embeddings to Qdrant.
- Unified model management under `models/` (VLM, Object Detection, Embedding, and ASR modules).
- Added orchestration script `main.py` to scan directories, extract metadata, and upload to Qdrant.
- Added `setup.sh` and `requirements.txt` scripts.
- Added `.gitignore`.

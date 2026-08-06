# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- OCR is now language-neutral by default: removed the Vietnamese-specialized Vintern ensemble and Vietnamese accent mapping. PP-OCRv6 preserves NFC-normalized recognized text, with `OCR_LANG=en` as the default and the fallback VLM reserved for low-confidence crops.

## [1.5.0] - 2026-08-04

### Changed
- Replaced `models/asr.py`'s `PhoWhisperASR` (`transformers.pipeline`, `vinai/PhoWhisper-large`) with `WhisperASR` (`faster-whisper`/CTranslate2, `deepdml/faster-whisper-large-v3-turbo-ct2` = Whisper large-v3-turbo) - VBS's V3C dataset isn't Vietnamese-centric, so the Vietnamese-specialized PhoWhisper model no longer fits; the 99-language multilingual Whisper large-v3-turbo replaces it. CTranslate2 has no Apple Silicon MPS backend, so device selection now uses `ctranslate2.get_cuda_device_count()` (cuda/cpu only).
- **Removed the ffmpeg-PATH workaround** added in an earlier round (see the note this superseded, below) - it existed only because `transformers`' ASR pipeline shells out to a bare `ffmpeg` binary. faster-whisper decodes audio via PyAV, which bundles the FFmpeg *libraries* directly in its wheel, so no `ffmpeg` binary on `PATH` is needed for ASR anymore. `AudioProcessor.extract_audio`'s own independent `bin/ffmpeg` fallback (unrelated, used for the initial video→WAV extraction) is unaffected.
- `AudioProcessor.transcribe_audio()` now drops low-confidence/silent/hallucinated segments (new `preprocessing/audio/asr_segment_filter.py`) before returning, and `preprocessing/main.py`'s speech payload gains `timestamp_end`, `words` (word-level timestamps), and `asr_avg_logprob`.

### Added
- `faster-whisper>=1.1.0` to `requirements.txt` (pulls `ctranslate2`, `av`, `onnxruntime`/bundled silero VAD).
- Config: `ASR_MODEL_ID`, `ASR_LANGUAGE`, `ASR_COMPUTE_TYPE`, `ASR_VAD_FILTER_ENABLED`, `ASR_WORD_TIMESTAMPS_ENABLED`, `ASR_MIN_AVG_LOGPROB`, `ASR_MAX_NO_SPEECH_PROB`, `ASR_MAX_COMPRESSION_RATIO` (replacing `PHOWHISPER_MODEL_ID`).

## [1.4.0] - 2026-07-16

### Changed
- Replaced VLM-based OCR with **PP-OCRv6** in `TextDetectorOCR` (`video/ocr.py`) - previously every "has text" frame (gated by an EasyOCR presence check, or all frames if EasyOCR wasn't installed) had its text extracted by a full VLM call; now PP-OCRv6 handles detection+recognition directly, and the VLM only re-reads individual crops recognized below `OCR_REC_SCORE_THRESHOLD` (default 0.5).
- Added an optional overlapping-tile OCR pass (`OCR_USE_TILING`, off by default) mirroring `ObjectDetector.detect_tiled()`'s pattern - written by hand rather than via `sahi`, since `sahi` 0.12.1's model registry has no `"paddleocr"` backend to plug into (`AutoDetectionModel.from_pretrained(model_type="paddleocr", ...)` isn't runnable as-is).
- Merged `ImageCaptioner.generate_temporal_caption()` + `extract_structured_attributes()` (two separate VLM calls per keyframe) into one `generate_frame_analysis()`/`generate_frame_analysis_batch()` call using a single unified JSON prompt (`UNIFIED_FRAME_PROMPT`) - `ocr_text` is intentionally excluded from this prompt since OCR now runs via PP-OCRv6, not the VLM.
- `main.py`'s per-scene keyframe loop now calls `generate_frame_analysis_batch()` once across all of a scene's keyframes instead of two VLM calls per keyframe in a loop.
- Replaced `select_diverse_keyframes()`'s fixed-threshold greedy diversity filter with **Adaptive Keyframe Sampling**: `compute_scene_variance()` encodes a scene's candidate frames with a lightweight CLIP model and measures embedding variance; `get_adaptive_budget()` maps that variance to a keyframe budget (1 for static scenes, up to `KEYFRAME_MAX_BUDGET`=8 for dynamic ones); `select_diverse_keyframes()` then runs farthest-point sampling down to that budget using the real Qwen embedding space.
- `OpenAIVLM.generate_batch()` now issues requests concurrently via `ThreadPoolExecutor` (`VLM_BATCH_CONCURRENCY`, default 4) instead of a sequential loop - the OpenAI-compatible API has no native batch endpoint, so batching benefit only comes from the server (e.g. vLLM's continuous batching) seeing multiple concurrent in-flight requests.
- `QwenVLM.generate_batch()` now builds one batch of chat-template inputs and runs a single `model.generate()` call across all images, instead of looping `generate()` per image.

### Added
- `host_vllm.sh` (repo root, not `preprocessing/` - shared with `inference-code/` since both point at it via the same `models/openai_vlm.py` client) - self-hosts the local VLM via vLLM's OpenAI-compatible server for batch inference, mirroring `host_qdrant.sh`'s style. Requires an NVIDIA/AMD GPU - vLLM doesn't run on Apple Silicon or CPU-only machines, so this is meant for the actual GPU/competition server, not local dev.
- `models/clip_embedder.py` (`LightweightCLIPEmbedder`) - CLIP ViT-B/32 wrapper used only for Adaptive Keyframe Sampling's scene-variance step, not the indexing embedding space.
- Config: `OCR_LANG`, `OCR_REC_SCORE_THRESHOLD`, `OCR_USE_TILING`, `KEYFRAME_VARIANCE_LOW`/`KEYFRAME_VARIANCE_MID`/`KEYFRAME_MAX_BUDGET`, `VLM_BATCH_CONCURRENCY`, `DASHSCOPE_EMBEDDING_MODEL_NAME`.
- `paddleocr`/`paddlepaddle` (PP-OCRv6) and `clip`/`ftfy`/`regex` (lightweight CLIP) added to `requirements.txt`.

### Fixed
- `select_diverse_keyframes()` crashed with `ValueError: The truth value of an array with more than one element is ambiguous` when removing a selected frame from the remaining candidates via `list.remove()` - dict equality compared the numpy `"embed"` arrays elementwise; fixed by removing by index instead.
- `video/scene_detector.py`: a stray `import cv2` inside `detect_scenes()`'s fallback branch shadowed the module-level import for the whole function, causing `UnboundLocalError` on any `cv2` use earlier in the function.

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

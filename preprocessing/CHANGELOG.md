# Changelog

All notable changes to this project will be documented in this file.

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

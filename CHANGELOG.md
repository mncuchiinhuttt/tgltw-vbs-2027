# Changelog

All notable changes to the Multimedia Retrieval project will be documented in this file.

## [1.3.0] - 2026-07-16

### Changed
- Replaced VLM-based OCR with **PP-OCRv6** (detection + recognition) in `preprocessing/video/ocr.py` - the VLM is now only used to re-read individual crops PP-OCRv6 recognized with low confidence (`OCR_REC_SCORE_THRESHOLD`), not run on every frame.
- Merged the per-keyframe `temporal_caption` and `structured_attrs` VLM calls into a single unified JSON call (`ImageCaptioner.generate_frame_analysis`/`generate_frame_analysis_batch`), batched across a scene's keyframes in one `generate_batch()` call.
- `OpenAIVLM.generate_batch()` now issues requests concurrently (`ThreadPoolExecutor`, `VLM_BATCH_CONCURRENCY`) instead of sequentially, and `QwenVLM.generate_batch()` now runs one true batched `model.generate()` call across images instead of looping - both needed to get any real throughput benefit from a batch-serving backend (e.g. a self-hosted vLLM server).
- Replaced fixed-threshold greedy keyframe selection with **Adaptive Keyframe Sampling**: a lightweight CLIP (ViT-B/32) pass estimates per-scene visual variance to set a keyframe budget (1-8), then farthest-point sampling picks that many frames from the real Qwen embedding space.

### Added
- `host_vllm.sh` (repo root, shared by `preprocessing/` and `inference-code/`) - self-hosts the local VLM via vLLM's OpenAI-compatible server for batch inference (requires an NVIDIA/AMD GPU; does not run on Apple Silicon or CPU-only machines).
- `models/clip_embedder.py` (`LightweightCLIPEmbedder`) - used only for Adaptive Keyframe Sampling's scene-variance step, not the indexing embedding space.
- `DASHSCOPE_EMBEDDING_MODEL_NAME` config so `DashScopeCloudEmbedder`'s model can be swapped without code changes.

## [1.2.0] - 2026-07-09

### Changed
- Swapped the object detector from NVIDIA LocateAnything-3B to **YOLOE-26** across `models/`, `preprocessing/`, and `inference-code/` - open-vocabulary and text-prompted like before, but NMS-free end-to-end and runs natively on CUDA/MPS/CPU with no custom kernel requirements.
- Added an `EMBEDDING_OPTION` toggle (`"local"`/`"cloud"`) and a `DashScopeCloudEmbedder`, letting the visual embedding model run against a cloud API instead of loading an 8B model locally when running several large local models at once exceeds available memory.
- `OpenAIVLM` now supports a configurable `base_url`/model name, so any OpenAI-compatible provider (e.g. QwenCloud) can be used for VLM calls, not just OpenAI directly.
- Updated the root README's shared-models overview to document these new options and correct several stale claims (embedding dimensions were documented as 1536d/512d; the actual values are 4096d/768d).

### Fixed
- `webapp/backend/main.py`: fixed a module-name collision where `from main import load_vlm` resolved to the backend's own module (uvicorn loads it as `"main"`) instead of `inference-code/main.py`, raising an `ImportError` on every search request; replaced with local `load_vlm()`/`load_embedder()` helpers.
- The webapp dev server's `reload=True` now explicitly watches `models/`, `preprocessing/`, and `inference-code/` (`reload_dirs`) - previously it only watched `webapp/backend/`, so edits to shared code silently kept running stale until a full process restart.

## [1.1.0] - 2026-07-02
### Added
- **Interactive WebApp**:
  - Implemented a FastAPI backend serving search query results, on-the-fly video frame extraction using OpenCV, video range stream hosting, and subprocess execution monitoring.
  - Implemented a React frontend (Vite + TS + React Router) configured with the team name `"The Gays Lead The World" from RMIT University Vietnam`.
  - Implemented **futuristic cyber-tech Light Mode** theme with grid blueprints, hovering glow border highlights, and scanning animation overlays.
  - Integrated custom shadcn/ui primitives (`Card`, `Badge`, `Dialog`, `Progress`, and `@radix-ui/react-select` based `Select`).
- **Batch Query Execution**:
  - Added `queries/` registry folder with a default `queries.json` template.
  - Implemented a standalone CLI script `inference-code/batch_query.py` to run search queries in batch.
  - Added an interactive **Batch Queries Dashboard** to the webapp to trigger batch runs, tail logs, list outputs in a detailed grid, and play target segments instantly.
- **Project Runners**:
  - Added `run_webapp.py` (Python) and `run_webapp.sh` (Shell script) to clean ports, manage node dependencies, resolve virtual environments, and start frontend + backend dev servers concurrently.

### Changed
- Updated root `README.md` to document webapp launching commands and batch querying.

---

## [1.0.0] - 2026-07-01
### Added
- Initial pipeline scripts (`preprocessing/main.py`, `inference-code/main.py`).
- Models wrapper wrappers (`models/`).
- Qdrant hosting script.

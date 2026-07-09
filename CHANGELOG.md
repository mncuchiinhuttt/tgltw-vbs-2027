# Changelog

All notable changes to the Multimedia Retrieval project will be documented in this file.

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

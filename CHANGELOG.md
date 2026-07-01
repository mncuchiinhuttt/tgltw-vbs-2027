# Changelog

All notable changes to the Multimedia Retrieval project will be documented in this file.

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

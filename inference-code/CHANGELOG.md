# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-07-01

### Added
- Created the complete inference search engine pipeline.
- Implemented CQR (Conversational Query Rewriting) contextual query rewriting using VLMs.
- Implemented HyDE (Hypothetical Document Embeddings) to generate hypothetical frame descriptions.
- Integrated Qdrant Dense Vector Search + Sparse Match (via payload `text_blob`) and Reciprocal Rank Fusion (RRF).
- Created Type 1 (Textual-KIS) reranking, outputting `<video_name>, <timestamp>`.
- Created Type 2 (VQA) decomposition and bounding box crop-reranking (using Rex-Omni), returning `<video_name>, <timestamp>, <answer>`.
- Created Type 3 (Temporal-Alignment) sequence reasoning checking order/continuity, returning `<video_name>, <frame_id_1>, ..., <frame_id_n>`.
- Refactored all model loaders (`base_vlm.py`, `qwen_vlm.py`, `openai_vlm.py`, `embedding.py`, `object_detector.py`) to the shared root `models/` directory.
- Added `requirements.txt` and `.env` configuration file support.
- Added `README.md` guide for inference usage.

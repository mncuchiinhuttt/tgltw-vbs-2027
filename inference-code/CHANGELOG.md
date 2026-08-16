# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed - candidate pooling for a wider index
Preprocessing now indexes several frames per shot rather than one, which changes what a fixed-size candidate pool contains. `diversify_by_scene` collapses each scene to a single hit, so a pool of near-duplicate frames of one strongly-matching moment would leave the reranker with fewer distinct events than before the index grew.

- `TOP_K_RETRIEVAL` raised to 200, and dense hits are capped at `DENSE_MAX_PER_SCENE` per (video, scene) so no single moment can monopolise the pool.
- `sparse_search` has its own `SPARSE_TOP_K_RETRIEVAL` depth. It reads payload matches through Qdrant `scroll`, which returns points in ID order rather than by relevance, so its rank signal is arbitrary beyond "matched the filter" - deepening it in step with the dense branch would only pour more noise into the RRF fusion.
- `get_all_points_for_video` is paginated. It previously assumed a video held "at most a few hundred" keyframes and truncated at a single scroll call, which would now silently hide the tail of a long video from TRAKE's alignment and from in-video refinement.

### Added
- `dense_search_regions`: searches the region crops indexed under `modality="region"`, collapsing several crops of one frame to its best-scoring crop and returning the **parent frame's** payload under the parent's point id. Regions never appear as results in their own right, so they cannot evict their own parent from the diversified grid or add duplicate frame indices to TRAKE's timeline. Returns `[]` when region indexing was never run.
- Query-time keyframe extraction inside `in_video_refine` (`QUERY_TIME_EXTRACTION_ENABLED`, off by default; needs `VIDEO_SOURCE_DIR` readable from the query host). Every other step here reorders frames that are already indexed and so cannot recover a moment offline selection dropped; this decodes the video and scores frames that have no index point at all. Extracted frames carry a synthetic `query-time:` id and are never fed back to Qdrant.

## [1.3.0] - 2026-07-16

### Changed
- `OpenAIVLM.generate_batch()` (shared `models/`) now issues requests concurrently via `ThreadPoolExecutor` instead of a sequential loop, and `QwenVLM.generate_batch()` now runs one true batched `model.generate()` call across images - enables real throughput benefit from a self-hosted vLLM backend for any inference-code flow that batches VLM calls.
- `DashScopeCloudEmbedder`'s model name is now configurable via `DASHSCOPE_EMBEDDING_MODEL_NAME` instead of hardcoded to `tongyi-embedding-vision-plus`.

### Added
- `VLM_BATCH_CONCURRENCY` config controlling how many concurrent requests `OpenAIVLM.generate_batch()` issues.
- `host_vllm.sh` (repo root, shared with `preprocessing/`) - self-hosts the local VLM via vLLM's OpenAI-compatible server for batch inference (requires an NVIDIA/AMD GPU).

## [1.2.0] - 2026-07-09

### Changed
- Swapped zero-shot object detector from NVIDIA LocateAnything-3B to **YOLOE-26** across search configuration and model loaders, matching the same swap in `preprocessing/`.
- Added `EMBEDDING_OPTION` (`"local"`/`"cloud"`) support via a new `load_embedder()` helper in `main.py`, `batch_query.py`, and the webapp backend, mirroring the existing `load_vlm()` pattern; supports `DashScopeCloudEmbedder` as a temporary lower-memory alternative to the local `QwenVL8BEmbedder`.
- `OpenAIVLM` now supports a configurable `base_url`/model name for use with alternative OpenAI-compatible providers (e.g. QwenCloud) instead of only OpenAI directly.

### Fixed
- Added missing `DETECTOR_OPTION` config value.
- Migrated Qdrant client calls from the removed `QdrantClient.search()` method to `query_points()`, matching the installed `qdrant-client` 1.18 API.
- Created `inference-code/.env` (was missing entirely - search/webapp queries were silently falling back to mismatched defaults, including a different embedding model than what the index was actually built with).

## [1.1.0] - 2026-07-05

### Changed
- Swapped zero-shot object detector from Rex-Omni to **NVIDIA LocateAnything-3B** (`nvidia/LocateAnything-3B`) across the search configurations and model loaders.

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

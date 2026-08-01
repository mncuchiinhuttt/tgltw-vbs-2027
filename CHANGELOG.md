# Changelog

All notable changes to the Multimedia Retrieval project will be documented in this file.

## [1.9.0] - 2026-08-02

### Added
- **`preprocessing/official_assets.py`**: optional loaders for the AIC dataset assets BTC provides alongside raw videos (`Thong tin vong So tuyen AIC2026.pdf`, section 3) - `Objects/` (Faster R-CNN/OpenImages V4 detections), `Metadata/` (YouTube title/description JSON), and a keyframe-index map used to nearest-match our own extracted keyframes (from scene-detection + AKS) to BTC's own keyframe numbering. Every lookup gracefully returns empty when a video has no matching official file, so this has zero effect on datasets that don't ship these assets.
  - `preprocessing/main.py`: merges official Objects detections into `detected_objects` (IoU-deduped against YOLOE+SAM3, denormalized against the actual keyframe size) for whichever official keyframe is nearest (by `frame_idx`) to ours; folds official Metadata title/description into `text_blob` and a new `video_metadata` payload field.
  - **Not wired in this PR**: the CLIP-features (`clip-ViT-B-32`) `.npy` loader exists (`load_official_clip_feature`) but isn't used to replace `LightweightCLIPEmbedder`'s scene-variance step - BTC's official keyframes are much sparser than our own per-scene Adaptive Keyframe Sampling candidates, so most candidate frames wouldn't have a close-enough official match to substitute; left as a documented follow-up rather than forcing an awkward partial integration.
  - **Verify against real data before fully trusting**: no real downloaded sample was available when this was written - field/path names are reconstructed from the PDF's prose description (see the module's own docstring for exactly which assumptions are riskiest, especially the keyframe-index-map file format, which the PDF never actually names).

Source: `Thong tin vong So tuyen AIC2026.pdf` - see Notion "Long Note Aug 2 2026" (item 5) for the full list of pipeline/competition-rule misalignments this addresses.

## [1.8.0] - 2026-08-02

### Changed
- **TRAKE (Type 3) reranking replaced with DANTE-inspired DP alignment** (`Reranker.rerank_type3_temporal()`, `inference-code/search/reranker.py`) - previously scored an entire candidate sequence with one holistic VLM call; now runs the competition's own two-stage design (`Thong tin vong So tuyen AIC2026.pdf`):
  - **Stage 1 (Retrieval)**: ranks candidate videos by their best frame-hit's RRF score (generalized to the top `TRAKE_MAX_VIDEOS_TO_ALIGN` videos rather than exactly one, so the AIC scoring rule's up-to-100-ranked-answers allowance still applies to TRAKE).
  - **Stage 2 (Alignment)**: `QueryProcessor.decompose_temporal_events()` splits the query into an ordered list of sub-events; `HybridSearcher.get_all_points_for_video()` fetches the video's *entire* frame timeline (not just whatever made the initial candidate pool); a new dynamic-programming subsequence alignment (`_align_events_dp`, O(N events x M frames)) picks the best chronologically-ordered frame per sub-event, maximizing total cosine similarity. Verified against brute-force search over 200 randomized small inputs - exact match every time.
  - Reference: DANTE in *"Integrated Semantic and Temporal Alignment for Interactive Video Retrieval"* (arXiv:2512.13169, AIC 2025).

Source: `Thong tin vong So tuyen AIC2026.pdf` + AIC 2024/2025 team research - see Notion "Long Note Aug 2 2026" (item 4) for the full list of pipeline/competition-rule misalignments this addresses.

## [1.7.0] - 2026-08-02

### Added
- **Ranked-list submission output (up to 100 answers/query)**: the AIC scoring rule averages `R@1/5/20/50/100` (best score within each rank prefix) - submitting only 1 answer per query wastes the credit available at higher k. `SUBMISSION_TOP_K` (default 100) widens retrieval/diversification; `RERANK_TOP_K` (default 20) scopes the expensive VLM rerank pass to the head of that pool, with the rest appended in original retrieval-rank order (`reranker.rerank_with_tail()`, `inference-code/search/reranker.py`) rather than paying for a VLM call per candidate just to rank the tail. `rerank_type3_temporal` (TRAKE) needs no head/tail split - it VLM-scores once per distinct candidate video, not per frame.
- `inference-code/batch_query.py` now writes a real ranked submission file, `batch_submission.csv` (one row per `(query, rank, answer)` up to `SUBMISSION_TOP_K`) - the existing `batch_results.json`/`batch_results.csv` (rank-1 only) are kept unchanged for backward compatibility with the webapp's batch results table.
- `inference-code/main.py` (CLI) and `evaluation/run_eval.py` updated to the same widened-pool/`rerank_with_tail` behavior, so a manual query and the eval harness both reflect what `batch_query.py` actually submits.
- `webapp/backend/main.py` (`/api/search`) intentionally left unchanged - it's an interactive human-facing search UI, not the competition submission path, and widening it to 100 candidates per click would slow down live search for no scoring benefit.

Source: `Thong tin vong So tuyen AIC2026.pdf` (AIC 2026 Sơ tuyển rules) - see Notion "Long Note Aug 2 2026" (item 1) for the full list of pipeline/competition-rule misalignments this addresses.

## [1.6.0] - 2026-08-02

### Fixed
- **Type 3 (TRAKE) `frame_ids` were Qdrant point UUIDs, not real video frame indices**: `Reranker.rerank_type3_temporal()` returned `[f["id"] for f in sorted_frames]` - a random UUID assigned at index time, carrying no temporal/frame-position meaning. Fixed to use the frame's real native video frame index (`payload["frame_idx"]`), matching the AIC competition's actual `<video_id>, <frame_id_1>, ..., <frame_id_N>` answer format.
- **Type 1/2 output used `timestamp` (seconds) instead of `frame_id` (frame index)**: the AIC competition's answer format is `<video_id>, <frame_id>` where `frame_id` is a native video frame index, not a timestamp in seconds. `inference-code/main.py`, `inference-code/batch_query.py` now report `payload["frame_idx"]` (falling back to timestamp with a warning for older indexed points that predate this field).
- **TRAKE ground-truth tolerance was ~10x too loose**: the competition's per-event semantic-keyframe window is documented as usually under 10 frames (a fraction of a second at typical fps), while `evaluation/run_eval.py` used a flat 3-second tolerance for all query types. Added `FRAME_MATCH_TOLERANCE` (frame-based, default 5) preferred over `TIMESTAMP_TOLERANCE_SEC` whenever `frame_id` is present in `ground_truth`/`event_frames`.

### Added
- `preprocessing/main.py`: every video keyframe payload now carries `frame_idx` (the native video frame index already computed by `extract_candidate_frames`, previously computed but discarded before this fix).
- `evaluation/eval_queries.json` / `evaluation/README.md`: ground truth schema now documents `frame_id` (preferred) alongside `timestamp` (fallback) for Type 1/2's target and Type 3's `event_frames`.

Source: `Thong tin vong So tuyen AIC2026.pdf` (AIC 2026 Sơ tuyển rules) - see Notion "Long Note Aug 2 2026" for the full list of pipeline/competition-rule misalignments this addresses (this is item 2+3 of that list).

## [1.5.0] - 2026-07-27

### Added
- **Segment-level Structured Events** (`preprocessing/video/captioner.py`): `ImageCaptioner.generate_scene_events()` synthesizes an `{actions, ordered_events}` list per scene from its already-computed per-frame captions + real timestamps (text-only VLM call, no extra image analysis) - helps Type 3 (Temporal-Alignment) skip inferring event order from prose captions alone. Stored in the Qdrant payload under `ordered_events`/`actions`, with `actions` also flattened into `text_blob` for BM25. `SCENE_NARRATIVE_PROMPT` is now capped to ~30-50 words so `scene_narrative` reads as a compact summary rather than a free-form paragraph.
- **Result Diversification** (`inference-code/search/hybrid_search.py`): `HybridSearcher.diversify_by_scene()` collapses candidates down to the best-scoring one per `(source_file, scene_id)` right after RRF fusion, before Stage 3 reranking, across all three query types - top-K no longer gets flooded by several near-duplicate keyframes from the same event at the expense of covering distinct events.
- Both features originate from Khoa's segment-centric retrieval proposal ("Khoa: Adaptive Sampling & Retrieval Accuracy") - adapted to layer on top of the existing frame-level index rather than replacing it, since Type 1/3 require an exact frame-level output.

## [1.4.0] - 2026-07-27

### Changed
- Object Detection and OCR now run **SAM3-gated**, region-restricted instead of full-frame:
  `RegionProposer` (`models/region_proposer.py`, `facebook/sam3` Promptable Concept Segmentation, zero-shot) proposes candidate regions from general concept prompts first - SAHI-style tiling (512x512, 0.2 overlap) and the actual detector/recognizer only run inside those regions, and the whole detection/OCR step is skipped for a keyframe if SAM3 finds no matching region at all.
  - `ObjectDetector.detect_in_regions()` (`models/object_detector.py`) replaces the previous full-frame `detect()` + fixed-label (`TILED_DETECTION_LABELS`, license plates only) `detect_tiled()` merge in `preprocessing/main.py`'s `detect_objects()` - tiling now applies uniformly to whatever SAM3 flags, not one hardcoded category. `detect()`/`detect_tiled()` themselves are unchanged and still used at query-time by `inference-code/search/reranker.py`.
  - `TextDetectorOCR` (`preprocessing/video/ocr.py`) is restructured into a two-stage pipeline: detection-only tiling within SAM3's proposed text/sign regions, then a separate recognition stage per surviving text-box crop. Crops shorter than `OCR_SR_MIN_HEIGHT_PX` (16px) are super-resolved (`SuperResolutionUpscaler`, `models/super_resolution.py`, Real-ESRGAN x4) before recognition. Recognition is now an **ensemble** of PP-OCRv6 and `VinternRecognizer` (`models/vintern_ocr.py`, Vintern-1B-v3.5) - highest-confidence result wins. If that's still below `OCR_REC_SCORE_THRESHOLD`, a dedicated lightweight fallback VLM (`SmolVLM2FallbackVLM`, `models/fallback_vlm.py`, SmolVLM2-500M-Video-Instruct) re-reads the crop - previously this escalation path used the much more expensive main captioning VLM.
  - Removed `OCR_USE_TILING` and `TILED_DETECTION_LABELS`/`_EN` (superseded by SAM3-gated region tiling, which now applies unconditionally). Added `SAM3_MODEL_ID`, `REGION_PROPOSAL_CONF_THRESHOLD`, `SAHI_TILE_SIZE`/`SAHI_TILE_OVERLAP`, `OBJECT_REGION_CONCEPTS_EN`, `OCR_REGION_CONCEPTS_EN`, `OCR_SR_MIN_HEIGHT_PX`, `REAL_ESRGAN_MODEL_ID`, `VINTERN_MODEL_ID`, `FALLBACK_VLM_MODEL_ID` in `preprocessing/config.py`.

### Added
- New Qdrant payload field `detected_text` - structured per-box OCR results (`{bbox, text, conf, accentless_text, source}`) alongside the existing flattened `ocr_text` string (kept for backward compatibility with `inference-code/search/reranker.py`, `evaluation/run_eval.py`, and the webapp frontend).
- `download_assets.py` now also fetches SAM3 (gated - requires accepting the license at https://huggingface.co/facebook/sam3 and setting `HF_TOKEN`), Vintern-1B-v3.5, the fallback VLM, and Real-ESRGAN weights.

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

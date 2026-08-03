# VBS 2027 - Multimedia Retrieval System

This repository is a full-history clone of our HCMC AI Challenge 2026 pipeline, adapted for the **Video Browser Showdown (VBS) 2027** competition. See [`VBS_GUIDE.md`](VBS_GUIDE.md) for the full competition reference (task types, scoring, DRES, datasets).

VBS is **live/interactive**: an operator drives searches by hand under a 5-7 minute per-task clock, so the system prioritizes low-latency, iterative refinement (relevance feedback, query-by-example, temporal window search) over the batch/offline, ranked-list-of-100 style used for AIC's Sơ tuyển. The underlying preprocessing/indexing pipeline (multimodal embeddings, Qdrant index, VLM/OCR/ASR models) is shared and reused as-is; the retrieval/webapp layer has been reworked for interactive use — see the [WebApp Dashboard](#5-webapp-dashboard--interactive-session) section below.

The system processes raw datasets (video, images, and audio), generates multimodal embeddings, indexes them to a Qdrant database, and retrieves matching frames/sequences for three query types: Textual-KIS (Type 1), VQA (Type 2), and Temporal-Alignment (Type 3).

---

## Directory Structure

```
tgltw-vbs-2027/
├── README.md              # Global workspace documentation
├── VBS_GUIDE.md           # VBS 2027 competition reference (tasks, scoring, DRES, datasets)
├── .env.template          # Root-level secrets template (HF_TOKEN, DRES_* credentials)
├── .gitignore             # Root git ignore (excludes /weights/, /datasets/, webapp/backend/logs/)
├── download_assets.py     # Script to automate downloading weights from Hugging Face
├── host_vllm.sh           # Self-hosts the local VLM via vLLM for batch inference (GPU only) - shared by preprocessing/ and inference-code/
├── evaluation/            # Standalone evaluation & benchmarking module
│   ├── README.md          # Evaluation module documentation & instructions
│   ├── eval_queries.json  # Annotated example query set with ground_truth (fill in real data)
│   ├── requirements.txt   # Optional deps (ragas, datasets) for generation-quality metrics
│   └── run_eval.py        # Standalone benchmark runner (Latency, Recall@K, MRR, optional Ragas)
├── models/                # [SHARED PYTHON MODELS] 
│   ├── base_vlm.py        # Abstract VLM interface
│   ├── qwen_vlm.py        # Local Qwen3-VL vision-language model loader
│   ├── openai_vlm.py      # OpenAI GPT 5.5 Pro API vision-language handler
│   ├── embedding.py       # QwenVL8BEmbedder, M2DClapEmbedder & DashScopeCloudEmbedder
│   ├── clip_embedder.py   # Lightweight CLIP embedder for keyframe scene-variance estimation
│   ├── asr.py             # faster-whisper (Whisper large-v3-turbo) speech-to-text transcriber
│   ├── object_detector.py # YOLOE-26 open-vocabulary object detector
│   ├── region_proposer.py  # SAM3 zero-shot region proposal (Object Detection + OCR pre-filter)
│   ├── super_resolution.py # Real-ESRGAN x4 conditional upscaling for small OCR crops
│   ├── vintern_ocr.py      # Vintern-1B-v3.5 OCR recognition ensemble member
│   └── fallback_vlm.py     # Lightweight SmolVLM2 fallback for low-confidence OCR crops
├── preprocessing/         # Dataset indexing pipeline (shared as-is with AIC)
│   ├── config.py          # Preprocessing settings, API URLs, and thresholds
│   ├── .env               # API Keys and model configurations (ignored)
│   ├── main.py            # Orchestrator to scan data, extract captions/embeddings
│   ├── requirements.txt   # Dependencies for preprocessing
│   ├── setup.sh           # Environment setup shell script
│   ├── host_qdrant.sh     # Starts Qdrant (via Docker or standalone binary download)
│   └── docker-compose.yml # Docker Compose config for Qdrant
├── inference-code/        # Retrieval and query engine
│   ├── config.py          # Search parameters, thresholds, and Qdrant settings (defaults tuned for VBS live latency)
│   ├── .env               # API Keys and model configurations (ignored)
│   ├── main.py            # CLI query parser for Type 1, 2, 3 retrieval
│   ├── search/hybrid_search.py # HybridSearcher - dense/sparse fusion, Rocchio feedback, temporal window match
│   └── requirements.txt   # Dependencies for inference
├── webapp/                # Interactive operator dashboard (VBS session UI + DRES integration)
│   ├── backend/           # FastAPI backend - see "WebApp Dashboard" section for endpoint list
│   │   ├── main.py            # API endpoints (search, feedback, DRES proxy, logging)
│   │   ├── dres_client.py     # Thin REST wrapper for the DRES evaluation server
│   │   ├── interaction_log.py # Local JSONL + best-effort DRES interaction/query logging
│   │   └── requirements.txt   # Backend dependencies (fastapi, requests, etc.)
│   └── frontend/          # React + Vite operator UI
│       └── src/components/
│           ├── ResultCard.tsx        # Result card: feedback, query-by-example, in-video search, DRES submit
│           └── BrowseVideoDialog.tsx # Full-video keyframe browser dialog
├── weights/               # [IGNORED] Downloaded model weights (.pth, .bin)
└── datasets/              # [IGNORED] Place raw videos, images, and audios here
```

---

## Shared Models Architecture

To avoid duplicate codebase wrappers, all model configurations and execution logic are stored in the root `models/` directory.

- **VLM backends** (`VLM_OPTION`): `local` (offline HuggingFace Qwen3-VL, `generate_batch()` runs one true batched `model.generate()` call) or `openai` (any OpenAI-compatible endpoint - OpenAI itself, another provider via `OPENAI_BASE_URL`/`OPENAI_VLM_MODEL_NAME` e.g. QwenCloud, or a self-hosted vLLM server for batch inference, see `host_vllm.sh` below). `generate_batch()` issues concurrent requests (`VLM_BATCH_CONCURRENCY`) so a batch-serving backend gets real throughput benefit instead of one request at a time. Used identically by both `preprocessing/` and `inference-code/` since both point at the same shared `models/openai_vlm.py` client.
- **Embeddings** (`EMBEDDING_OPTION`): `local` (`QwenVL8BEmbedder`, 4096d text/image space, ~15GB) or `cloud` (`DashScopeCloudEmbedder`, model configurable via `DASHSCOPE_EMBEDDING_MODEL_NAME`, no local weights - useful when running several large local models at once exceeds available memory). `M2DClapEmbedder` (768d sound space) is always local.
- **Object Detection**: `ObjectDetector` wraps YOLOE-26 (open-vocabulary, text-prompted, NMS-free end-to-end) to locate target objects, with tiled inference for small objects (`detect_tiled`), example-crop-based visual prompting (`detect_visual_prompt`), and SAM3-gated region-restricted tiling (`detect_in_regions`) used by the preprocessing pipeline (see below).
- **SAM3-gated Detection & OCR pre-filter**: `RegionProposer` (`models/region_proposer.py`) wraps `facebook/sam3` (Promptable Concept Segmentation, zero-shot, **gated on Hugging Face** - accept the license at https://huggingface.co/facebook/sam3 and set `HF_TOKEN` before downloading/running it) to propose candidate regions from general concept prompts ("human"/"vehicle"/"small distinct object" for Object Detection, "text or sign region" for OCR) before SAHI-style tiling (512x512, 0.2 overlap) and the actual detector/recognizer run - a keyframe's detection/OCR step is skipped entirely if SAM3 finds no matching region.
- **OCR recognition ensemble**: `preprocessing/video/ocr.py`'s `TextDetectorOCR` recognizes each text-box crop with both PP-OCRv6 and `VinternRecognizer` (`models/vintern_ocr.py`, Vintern-1B-v3.5), keeping whichever is more confident; crops shorter than 16px are upscaled first via `SuperResolutionUpscaler` (`models/super_resolution.py`, Real-ESRGAN x4), and crops where the ensemble's best confidence is still low fall back to a dedicated lightweight VLM (`SmolVLM2FallbackVLM`, `models/fallback_vlm.py`) rather than the main captioning VLM.
- **Adaptive Keyframe Sampling**: `models/clip_embedder.py`'s lightweight CLIP model estimates how visually static/dynamic a scene is, sizing a per-scene keyframe budget (1-8) before the real embedding model runs farthest-point sampling within it - see `preprocessing/video/scene_detector.py`.
- **ASR**: `WhisperASR` (`models/asr.py`, faster-whisper/CTranslate2 running Whisper large-v3-turbo, 99-language multilingual - VBS's V3C dataset isn't Vietnamese-centric like AIC's, so a general multilingual model replaces the AIC pipeline's Vietnamese-specialized PhoWhisper) transcribes speech with word-level timestamps; silero VAD skips non-speech regions, and `preprocessing/audio/asr_segment_filter.py` drops any remaining low-confidence/hallucinated segments (OpenAI Whisper's own reference thresholds on `avg_logprob`/`no_speech_prob`/`compression_ratio`) before they reach embedding + indexing.

The scripts dynamically append the workspace root to `sys.path` to import `models.*` from anywhere.

---

## Getting Started

### 0. Install the Python environment with uv (recommended)

The project uses one shared Python 3.12 environment at the repository root.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once, then sync only the dependencies needed for the workflow:

```bash
# Dashboard/backend only (fastest first run)
uv sync

# Retrieval/inference dependencies
uv sync --group inference

# Full indexing + OCR/SAM3 environment
uv sync --group preprocessing

# Optional upload/evaluation tools
uv sync --group upload
uv sync --group evaluation
```

Use `uv run ...` for every Python command; it automatically uses the locked
`.venv` and does not require manual activation. `uv.lock` is committed so
machines running the same project resolve the same dependency versions.

### 1. Download Model Checkpoints

Before running the download script, accept the SAM3 license at https://huggingface.co/facebook/sam3 (it's a gated repo) and set `HF_TOKEN` in your `.env` - the SAM3 download will fail without it.

Run the download script from the root folder to download weights for Whisper large-v3-turbo (faster-whisper/CTranslate2 format), CLAP, the YOLOE-26 detector, SAM3, Vintern-1B-v3.5, the fallback VLM, and Real-ESRGAN into the `weights/` folder:

```bash
uv run python download_assets.py
```

### 2. Host the Database (Qdrant)

Start the Qdrant server instance. The script will automatically try using Docker if installed, or download and run the native standalone Qdrant binary in the background:

```bash
cd preprocessing
chmod +x host_qdrant.sh
./host_qdrant.sh
```

Access the Qdrant Dashboard at: [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### 2b. Host the VLM via vLLM (optional, GPU required)

For batch inference throughput when self-hosting the local VLM (instead of one-request-at-a-time HuggingFace `transformers` calls), serve it through vLLM's OpenAI-compatible server - shared by both `preprocessing/` and `inference-code/`:

```bash
chmod +x host_vllm.sh
./host_vllm.sh
```

Requires an NVIDIA (CUDA) or AMD (ROCm) GPU - it does not run on Apple Silicon/macOS or CPU-only machines. Once the server is up, point either module's `.env` at it instead of loading a local HF model:

```bash
VLM_OPTION=openai
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_VLM_MODEL_NAME=<same model served by host_vllm.sh>
VLM_BATCH_CONCURRENCY=16   # raise this to actually use vLLM's continuous batching
```

### 2c. Migrating Indexed Data to a New Server

Everything Qdrant indexes (embeddings + payloads from `preprocessing/main.py`) lives on disk under `preprocessing/qdrant_storage/` - both `host_qdrant.sh`'s Docker path (`docker-compose.yml`'s `./qdrant_storage:/qdrant/storage` volume) and its standalone-binary fallback (`QDRANT__STORAGE__STORAGE_PATH=./qdrant_storage`) write there. Model weights (`weights/`) are **not** part of this - they're just re-downloaded via `download_assets.py` on the new server, no need to copy them.

**Option A - copy the storage directory directly (simplest, same Qdrant version only):**

```bash
# On the old server: stop Qdrant first so files aren't mid-write
cd preprocessing
docker compose down   # or: pkill qdrant   (if running the standalone binary)

# Copy the whole storage dir to the new server (any transfer tool works)
rsync -avz qdrant_storage/ new-server:/path/to/tgltw-vbs-2027/preprocessing/qdrant_storage/

# On the new server: start Qdrant as usual - it picks the copied data up automatically
cd preprocessing && ./host_qdrant.sh
```

Only safe when both servers run the **same Qdrant version** - `docker-compose.yml` pins `qdrant/qdrant:latest`, which can drift between two `docker compose up` runs on different machines/dates and silently change the on-disk storage format. Pin an explicit version tag (e.g. `qdrant/qdrant:v1.10.1`, matching `host_qdrant.sh`'s standalone-binary `QDRANT_VERSION`) in `docker-compose.yml` on both servers before relying on this option.

**Option B - Qdrant's snapshot API (recommended, version-tolerant, per-collection, no downtime on the source):**

```bash
# On the old server: snapshot each collection (visual_index, audio_env_index)
curl -X POST http://localhost:6333/collections/visual_index/snapshots

# List snapshots to get the exact filename just created
curl http://localhost:6333/collections/visual_index/snapshots

# Download it
curl -o visual_index.snapshot \
  http://localhost:6333/collections/visual_index/snapshots/<snapshot_name>

# Copy visual_index.snapshot to the new server, then restore it there
# (new server's Qdrant must be running first)
curl -X POST http://localhost:6333/collections/visual_index/snapshots/upload \
  -F "snapshot=@visual_index.snapshot"
```

Repeat per collection - this project uses two (`preprocessing/indexing/indexer.py`): `visual_index` (embeddings + all payloads: keyframes, OCR, objects, speech transcripts) and `audio_env_index` (CLAP ambient-audio embeddings). This is the officially supported migration path across Qdrant versions and doesn't require stopping the source server. See [Qdrant's snapshot docs](https://qdrant.tech/documentation/concepts/snapshots/) for restoring into a fresh collection name or a multi-node cluster instead.

---

## 3. Preprocessing & Indexing

1. Sync the preprocessing environment:
   ```bash
   uv sync --group preprocessing
   ```
2. Configure `.env` in `preprocessing/` (add API keys and choose model configurations).
3. Place your raw files in the global `datasets/` folder.
4. Run the pipeline:
   ```bash
   uv run --group preprocessing python preprocessing/main.py --data_dir datasets
   ```

### 3a. Reducing Index Size with Matryoshka (MRL) Truncation

`QwenVL8BEmbedder` (visual/text embeddings, default 4096-dim) is trained with Matryoshka
Representation Learning (arXiv:2601.04720): the leading N dimensions of its output vector are
themselves a valid, independently-meaningful embedding, not an arbitrary slice. Setting
`EMBEDDING_MRL_DIM` in `preprocessing/.env` (e.g. `EMBEDDING_MRL_DIM=1024`) truncates every
new vector to that many leading dims and re-normalizes it before indexing — shrinking Qdrant
storage and speeding up search roughly in proportion to the size reduction, at zero extra
inference cost (same model forward pass, just a shorter output kept). Does **not** apply to
`M2DClapEmbedder` (ambient-audio, 768-dim) — that model wasn't trained with MRL, so truncating
its output would break its meaning; `EMBEDDING_MRL_DIM` only affects the Qwen embedder.

**How much to cut:** general MRL results (and Qwen3's own embedding docs) suggest cutting to
1/4-1/8 of the full dimension (i.e. 1024-512 out of 4096) usually keeps accuracy close to the
untruncated vector, with degradation growing quickly below that. This is a starting point to
test, not a guarantee for this specific model/dataset — **validate empirically** before
committing to a value for a real competition index:

```bash
# Re-index a small sample (or re-embed the evaluation set) at a few candidate dims, then
# compare Recall@1/Recall@5/MRR for each against the untruncated baseline:
EMBEDDING_MRL_DIM=1024 uv run --group evaluation python evaluation/run_eval.py --output_file evaluation/eval_results_mrl1024.json
EMBEDDING_MRL_DIM=512  uv run --group evaluation python evaluation/run_eval.py --output_file evaluation/eval_results_mrl512.json
```

Pick the smallest dim whose Recall@K/MRR drop vs. the full 4096-dim baseline is negligible for
your query set — see `evaluation/README.md` for the eval runner's full docs, and
[`## 2c. Migrating Indexed Data to a New Server`](#2c-migrating-indexed-data-to-a-new-server)
if you change `EMBEDDING_MRL_DIM` after already indexing (existing vectors keep their old
dimension; changing this setting requires re-indexing, not just a config flip).

---

## 4. Query Retrieval (Inference)

1. Setup the environment:
   ```bash
   cd ../inference-code
   uv sync --group inference
   ```
2. Configure `.env` in `inference-code/` (point to Qdrant host and define query models).
3. Run search queries from CLI:

#### Type 1: Textual-KIS (Retrieves matching video name and timestamp)

```bash
uv run --group inference python inference-code/main.py --type 1 --query "một người đang lái xe máy đi qua ngã tư dưới trời mưa"
```

#### Type 2: VQA (Detects targets, crops local bounding boxes, scores via VLM, and answers)

```bash
uv run --group inference python inference-code/main.py --type 2 --query "người mặc áo đỏ đang dắt xe đạp màu xanh ở giây thứ mấy?"
```

#### Type 3: Temporal-Alignment (Reranks sequence of events chronologically)

```bash
uv run --group inference python inference-code/main.py --type 3 --query "đầu tiên có người chạy bộ qua đường, tiếp đến chiếc ô tô đen đi qua"
```

---

## 5. WebApp Dashboard & Interactive Session

The webapp is the operator's live console during a VBS task: single/batch queries, database stats, and the interactive session tools described below. Each operator runs their own backend instance (no multi-tenant session store) — this matches VBS's one-workstation-per-operator setup.

### Start the WebApp Dev Servers

Concurrently run both the React frontend and the FastAPI backend dev servers with:

```bash
# Python Runner (Resolves port conflicts & detects venv automatically)
uv run python run_webapp.py

# Bash Runner
./run_webapp.sh
```

- Open **Dashboard (Vite)**: [http://localhost:5173](http://localhost:5173)
- Open **API Docs (FastAPI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Interactive Session Endpoints

Beyond the base `/api/search`, `/api/status`, and media endpoints, the backend exposes the interactive tools an operator uses mid-task:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/feedback` | Rocchio-style relevance feedback (👍/👎 on results) — re-searches with an adjusted query vector |
| `POST /api/query-by-example` | Re-search using an already-indexed result's own stored vector, no re-embedding |
| `POST /api/search-by-image` | **KIS-V** — search by an uploaded photo/screenshot instead of text (embeds the upload, searches by that vector directly, no RRF fusion) |
| `POST /api/temporal-search` | **N-query temporal chain** search (`queries: string[]`, N≥2) — finds the best chronologically-ordered frame chain per video, one step per query, each within a frame window of the previous step (Exquisitor-inspired sequence-chain matching) |
| `GET /api/browse-video/{video_name}` | Full keyframe listing for a single video, for manual scrubbing |
| `POST /api/in-video-search` | Manually-triggered deep search restricted to one candidate video |
| `POST /api/dres/login` / `GET /api/dres/current-task` | DRES session login and current-task lookup (backend-proxied, credentials never reach the frontend) |
| `POST /api/dres/submit` | Submit an answer to DRES. Accepts optional `video_name`/`force` — resubmitting a video already submitted for the same `task_id` returns a 409 warning (not a hard block, overridable with `force: true`), since VBS's AVS scoring gives no extra credit for a duplicate video and penalizes wrong resubmissions |

Every search/feedback/query-by-example/temporal-search call is logged locally to `webapp/backend/logs/interaction_log.jsonl` first, then best-effort pushed to DRES if `DRES_*` env vars are configured — a DRES outage never blocks the operator's response (`webapp/backend/interaction_log.py`).

### Result Quality & Precision Controls

- **Explainability**: `/api/search` results include a `matched_via` field (e.g. `["query", "hyde"]`) showing which fusion source(s) surfaced each hit, plus OCR/scene-narrative evidence shown unconditionally on VQA answer cards — so the operator can judge trust before acting instead of seeing one opaque score.
- **Temporal coherence re-scoring**: candidates from the same video within a small frame window boost each other's score, so a real event isn't left fragmented across several marginal individually-scored frames.
- **KIS-C clarification**: when the top results spread across many unrelated videos with no clear winner (ambiguous query), the response includes a `clarification` field with one system-generated narrowing question — shown as an amber banner in the UI. Gated behind `AMBIGUITY_THRESHOLD` (default `0.7`) so the common unambiguous case pays no extra cost.
- **Escalate precision on-demand**: `/api/search` accepts optional `exact` (force exact/brute-force Qdrant search), `verify` (force verification reranking), and `hnsw_ef` (graduated HNSW search-time recall/latency tradeoff, a middle ground between the default and `exact`) — all `None`/unset by default, so behavior only changes when an operator explicitly opts in per-search (the frontend exposes `exact`/`verify` as two checkboxes; `hnsw_ef` is API-only for now).
- **AVS diversification**: `/api/search` collapses candidates to the single highest-scoring hit per video+scene (`diversify_by_scene`) right after temporal coherence re-scoring, so results aren't flooded by near-duplicate keyframes of the same event — directly serving AVS's diversity-across-videos scoring.

### DRES Configuration

Copy `.env.template` to `.env` at the repo root (or into `webapp/backend/`) and fill in the competition's DRES details once known:

```bash
DRES_BASE_URL=
DRES_USERNAME=
DRES_PASSWORD=
DRES_EVALUATION_ID=
```

These are unverified against a live DRES instance — confirm against the actual VBS 2027 DRES deployment before competition day.

---

## 6. Batch Queries (Process Multiple Queries)

### CLI Execution

To execute multiple queries in batch from the terminal:

1. Place your queries in the `queries/queries.json` file.
2. Run:

```bash
cd inference-code
python batch_query.py --query_file ../queries/queries.json --output_dir ../queries/
```

3. Batch outputs will be saved to `queries/batch_results.json` and `queries/batch_results.csv`.

### WebApp Dashboard Execution

1. Navigate to the **Process Multiple Queries** tab in the main console.
2. Click the **Process Multiple Queries** button to run batch inference.
3. Review logs live in the terminal output widget, and interact with the results list (including click-to-play support for matched segments).

---

## 7. System Evaluation & Benchmarking

We provide a standalone, decoupled evaluation runner to measure **End-to-End Latency** and **Accuracy Metrics** (Recall@K, MRR, and optionally Ragas Faithfulness/Answer Correctness/Context Recall) without altering production codebase files.

### Running Benchmarks via CLI

Run the evaluation script from the `method/` directory:

```bash
# (Optional) install Ragas to compute real generation-quality metrics
    uv sync --group evaluation

# Run benchmark against the annotated evaluation query set (evaluation/eval_queries.json)
uv run --group evaluation python evaluation/run_eval.py

# Run benchmark with a custom, annotated query file and dataset path
uv run --group evaluation python evaluation/run_eval.py --query_file evaluation/my_eval_set.json --dataset_dir datasets --output_file evaluation/eval_results.json
```

- **Output Metrics**: Evaluates **Recall@1**, **Recall@5**, **MRR**, **Latency Breakdown** (HyDE, Search, Rerank), and **QPS Throughput** across Type 1 (KIS), Type 2 (VQA), and Type 3 (Temporal) queries. Ragas-based generation metrics report `N/A` if `ragas` isn't installed, rather than a fabricated score.
- Accuracy metrics require a `ground_truth`-annotated query file — do not point `--query_file` at `queries/queries.json`, which is the production query registry and has no ground truth.
- Detailed results are printed to stdout and saved to `evaluation/eval_results.json`. See `evaluation/README.md` for complete documentation.

---

## License

This repository's code is licensed under the [MIT License](LICENSE). This covers only the code in this repo — it does **not** extend to third-party model weights downloaded via `download_assets.py` (Whisper large-v3-turbo, YOLOE-26, SAM3, Vintern-1B-v3.5, Real-ESRGAN, etc.), each of which carries its own license/usage terms. SAM3 in particular is a gated Hugging Face repo requiring separate license acceptance — see [Getting Started](#1-download-model-checkpoints).

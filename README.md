# HCMC AI Challenge 2026 - Multimedia Retrieval Pipeline

This repository contains the complete preprocessing, indexing, and inference system for HCMC AI Challenge 2026 (Group A) – Multimedia Retrieval.

The system processes raw datasets (video, images, and audio), generates multimodal embeddings, indexes them to a Qdrant database, and retrieves matching frames/sequences for three types of queries: Textual-KIS (Type 1), VQA (Type 2), and Temporal-Alignment (Type 3).

---

## Directory Structure

```
Method/
├── README.md              # Global workspace documentation
├── .gitignore             # Root git ignore (excludes /weights/ and /datasets/)
├── download_assets.py     # Script to automate downloading weights from Hugging Face
├── host_vllm.sh           # Self-hosts the local VLM via vLLM for batch inference (GPU only) - shared by preprocessing/ and inference-code/
├── evaluation/            # Standalone evaluation & benchmarking module
│   ├── README.md          # Evaluation module documentation & instructions
│   └── run_eval.py        # Standalone benchmark runner (Latency, Recall@K, MRR, Ragas)
├── models/                # [SHARED PYTHON MODELS] 
│   ├── base_vlm.py        # Abstract VLM interface
│   ├── qwen_vlm.py        # Local Qwen3-VL vision-language model loader
│   ├── openai_vlm.py      # OpenAI GPT 5.5 Pro API vision-language handler
│   ├── embedding.py       # QwenVL8BEmbedder, M2DClapEmbedder & DashScopeCloudEmbedder
│   ├── clip_embedder.py   # Lightweight CLIP embedder for keyframe scene-variance estimation
│   ├── asr.py             # PhoWhisper speech-to-text transcriber
│   └── object_detector.py # YOLOE-26 open-vocabulary object detector
├── preprocessing/         # Dataset indexing pipeline
│   ├── config.py          # Preprocessing settings, API URLs, and thresholds
│   ├── .env               # API Keys and model configurations (ignored)
│   ├── main.py            # Orchestrator to scan data, extract captions/embeddings
│   ├── requirements.txt   # Dependencies for preprocessing
│   ├── setup.sh           # Environment setup shell script
│   ├── host_qdrant.sh     # Starts Qdrant (via Docker or standalone binary download)
│   └── docker-compose.yml # Docker Compose config for Qdrant
├── inference-code/        # Retrieval and query engine
│   ├── config.py          # Search parameters, thresholds, and Qdrant settings
│   ├── .env               # API Keys and model configurations (ignored)
│   ├── main.py            # CLI query parser for Type 1, 2, 3 retrieval
│   └── requirements.txt   # Dependencies for inference
├── weights/               # [IGNORED] Downloaded model weights (.pth, .bin)
└── datasets/              # [IGNORED] Place raw videos, images, and audios here
```

---

## Shared Models Architecture

To avoid duplicate codebase wrappers, all model configurations and execution logic are stored in the root `models/` directory.

- **VLM backends** (`VLM_OPTION`): `local` (offline HuggingFace Qwen3-VL, `generate_batch()` runs one true batched `model.generate()` call) or `openai` (any OpenAI-compatible endpoint - OpenAI itself, another provider via `OPENAI_BASE_URL`/`OPENAI_VLM_MODEL_NAME` e.g. QwenCloud, or a self-hosted vLLM server for batch inference, see `host_vllm.sh` below). `generate_batch()` issues concurrent requests (`VLM_BATCH_CONCURRENCY`) so a batch-serving backend gets real throughput benefit instead of one request at a time. Used identically by both `preprocessing/` and `inference-code/` since both point at the same shared `models/openai_vlm.py` client.
- **Embeddings** (`EMBEDDING_OPTION`): `local` (`QwenVL8BEmbedder`, 4096d text/image space, ~15GB) or `cloud` (`DashScopeCloudEmbedder`, model configurable via `DASHSCOPE_EMBEDDING_MODEL_NAME`, no local weights - useful when running several large local models at once exceeds available memory). `M2DClapEmbedder` (768d sound space) is always local.
- **Object Detection**: `ObjectDetector` wraps YOLOE-26 (open-vocabulary, text-prompted, NMS-free end-to-end) to locate target objects, with optional tiled inference for small objects (`detect_tiled`) and example-crop-based visual prompting (`detect_visual_prompt`).
- **Adaptive Keyframe Sampling**: `models/clip_embedder.py`'s lightweight CLIP model estimates how visually static/dynamic a scene is, sizing a per-scene keyframe budget (1-8) before the real embedding model runs farthest-point sampling within it - see `preprocessing/video/scene_detector.py`.
- **ASR**: `PhoWhisperASR` for transcribing speech with timestamps.

The scripts dynamically append the workspace root to `sys.path` to import `models.*` from anywhere.

---

## Getting Started

### 1. Download Model Checkpoints

Run the download script from the root folder to download weights for PhoWhisper, CLAP, and the YOLOE-26 detector into the `weights/` folder:

```bash
python download_assets.py
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

---

## 3. Preprocessing & Indexing

1. Setup the environment and activate it:
   ```bash
   cd preprocessing
   chmod +x setup.sh
   ./setup.sh
   source venv/bin/activate
   ```
2. Configure `.env` in `preprocessing/` (add API keys and choose model configurations).
3. Place your raw files in the global `datasets/` folder.
4. Run the pipeline:
   ```bash
   python main.py --data_dir ../datasets
   ```

---

## 4. Query Retrieval (Inference)

1. Setup the environment:
   ```bash
   cd ../inference-code
   pip install -r requirements.txt
   ```
2. Configure `.env` in `inference-code/` (point to Qdrant host and define query models).
3. Run search queries from CLI:

#### Type 1: Textual-KIS (Retrieves matching video name and timestamp)

```bash
python main.py --type 1 --query "một người đang lái xe máy đi qua ngã tư dưới trời mưa"
```

#### Type 2: VQA (Detects targets, crops local bounding boxes, scores via VLM, and answers)

```bash
python main.py --type 2 --query "người mặc áo đỏ đang dắt xe đạp màu xanh ở giây thứ mấy?"
```

#### Type 3: Temporal-Alignment (Reranks sequence of events chronologically)

```bash
python main.py --type 3 --query "đầu tiên có người chạy bộ qua đường, tiếp đến chiếc ô tô đen đi qua"
```

---

## 5. WebApp Dashboard

We provide a futuristic Light Mode dashboard for running single queries, batch queries, checking database statistics, and tracking logs.

### Start the WebApp Dev Servers

Concurrently run both the React frontend and the FastAPI backend dev servers with:

```bash
# Python Runner (Resolves port conflicts & detects venv automatically)
python3 run_webapp.py

# Bash Runner
./run_webapp.sh
```

- Open **Dashboard (Vite)**: [http://localhost:5173](http://localhost:5173)
- Open **API Docs (FastAPI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

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

We provide a standalone, decoupled evaluation runner to measure **End-to-End Latency** and **Accuracy Metrics** (Recall@K, MRR, Ragas Faithfulness) without altering production codebase files.

### Running Benchmarks via CLI

Run the evaluation script from the `method/` directory:

```bash
# Run benchmark with default test queries
python evaluation/run_eval.py

# Run benchmark with custom query and dataset paths
python evaluation/run_eval.py --query_file queries/queries.json --dataset_dir datasets --output_file evaluation/eval_results.json
```

- **Output Metrics**: Evaluates **Recall@1**, **Recall@5**, **MRR**, **Latency Breakdown** (HyDE, Search, Rerank), and **QPS Throughput** across Type 1 (KIS), Type 2 (VQA), and Type 3 (Temporal) queries.
- Detailed results are printed to stdout and saved to `evaluation/eval_results.json`. See `evaluation/README.md` for complete documentation.

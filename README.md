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
├── models/                # [SHARED PYTHON MODELS] 
│   ├── base_vlm.py        # Abstract VLM interface
│   ├── qwen_vlm.py        # Local Qwen3-VL vision-language model loader
│   ├── openai_vlm.py      # OpenAI GPT 5.5 Pro API vision-language handler
│   ├── embedding.py       # QwenVL8BEmbedder, M2DClapEmbedder & DashScopeCloudEmbedder
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

- **VLM backends** (`VLM_OPTION`): `local` (offline HuggingFace Qwen3-VL) or `openai` (any OpenAI-compatible endpoint - OpenAI itself, or another provider via `OPENAI_BASE_URL`/`OPENAI_VLM_MODEL_NAME`, e.g. QwenCloud's DashScope-compatible API).
- **Embeddings** (`EMBEDDING_OPTION`): `local` (`QwenVL8BEmbedder`, 4096d text/image space, ~15GB) or `cloud` (`DashScopeCloudEmbedder`, 1152d, no local weights - useful when running several large local models at once exceeds available memory). `M2DClapEmbedder` (768d sound space) is always local.
- **Object Detection**: `ObjectDetector` wraps YOLOE-26 (open-vocabulary, text-prompted, NMS-free end-to-end) to locate target objects, with optional tiled inference for small objects (`detect_tiled`) and example-crop-based visual prompting (`detect_visual_prompt`).
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

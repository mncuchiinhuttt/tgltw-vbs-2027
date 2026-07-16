# Inference Engine (HCMC AI Challenge 2026)

This module handles executing search queries against the Qdrant database using Hybrid Search, CQR/HyDE enhancements, and advanced reranking pipelines (Type 1 KIS, Type 2 VQA with bounding box crops, and Type 3 Temporal Alignment).

## Project Structure

```
inference-code/
├── config.py              # Configuration settings and backend options
├── main.py                # CLI entrypoint for search queries
├── requirements.txt       # Python dependencies
├── .env                   # Configuration overrides (API Keys, hosts)
├── .env.template          # Public template for env vars
└── search/
    ├── query_processor.py # CQR contextual rewrites and HyDE document generators
    ├── hybrid_search.py   # Qdrant Dense + Sparse (payload search) merged via RRF
    └── reranker.py        # VQA bbox cropping and temporal ordering verifications
```

*(Note: Shared models logic has been moved to the root `/models/` directory).*

## Features

1. **CQR & HyDE Query Enhancement**: Prompts a VLM to rewrite queries using conversational history and generates hypothetical frame descriptions to boost embedding search recall.
2. **Dense & Sparse Hybrid Search**: Retrieves candidates from Qdrant using both dense visual/text embeddings and sparse payload match queries, merging results via **RRF (Reciprocal Rank Fusion)**.
3. **Type 1 (Textual-KIS)**: Reranks retrieved candidate frames using a VLM to score frames against the search query, returning `<video_name>, <timestamp>`.
4. **Type 2 (Visual Question Answering)**: Decomposes queries into sub-queries, executes YOLOE-26 object detection on candidates, crops bounding boxes, scores VQA crops using a VLM, and returns `<video_name>, <timestamp>, <answer>`.
5. **Type 3 (Temporal-Alignment)**: Groups frames chronologically by video and evaluates the temporal correctness of sequence descriptions via a VLM, returning `<video_name>, <frame_id_1>, ..., <frame_id_n>`.

## Setup

Install requirements:
```bash
pip install -r requirements.txt
```

Set up your `.env` variables (e.g. `OPENAI_API_KEY`, `QDRANT_HOST`, `VLM_OPTION`).

For batch/concurrent VLM inference (e.g. when running `batch_query.py` against many queries), point `OPENAI_BASE_URL` at a self-hosted vLLM server instead of loading a local HF model - see `host_vllm.sh` at the repo root (GPU required) and raise `VLM_BATCH_CONCURRENCY` to actually use its continuous batching.

## Usage

Run queries using the CLI:

### Type 1: Textual-KIS
```bash
python main.py --type 1 --query "một người đang đạp xe đạp trên đường phố buổi sáng"
```

### Type 2: VQA (Visual Question Answering)
```bash
python main.py --type 2 --query "chiếc xe máy màu đỏ đỗ cạnh cây xăng có biển số là gì?" --dataset_dir /path/to/frames
```

### Type 3: Temporal-Alignment
```bash
python main.py --type 3 --query "đầu tiên người đó mở cửa xe, sau đó bước vào và lái đi"
```

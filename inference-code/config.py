import os
from dotenv import load_dotenv

# Load env variables from .env file if it exists
load_dotenv()

# API Keys & Endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Leave blank for OpenAI's default endpoint, or point at any OpenAI-compatible
# provider (e.g. QwenCloud's DashScope-compatible endpoint) to use a different
# vision-capable model without loading a local VLM
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
OPENAI_VLM_MODEL_NAME = os.getenv("OPENAI_VLM_MODEL_NAME", "gpt-5.5-pro")
# How many OpenAIVLM.generate_batch() requests to issue concurrently. Raise
# this when OPENAI_BASE_URL points at a self-hosted vLLM server (see
# host_vllm.sh at the repo root) to get its continuous-batching throughput benefit.
VLM_BATCH_CONCURRENCY = int(os.getenv("VLM_BATCH_CONCURRENCY", 4))

# Model configuration options
# Options: "local" (uses Qwen3-VL locally) or "openai" (uses GPT 5.5 Pro / GPT-4o style API)
VLM_OPTION = os.getenv("VLM_OPTION", "openai")
# Options: "local" (QwenVL8BEmbedder, ~15GB) or "cloud" (DashScopeCloudEmbedder,
# via OPENAI_API_KEY/OPENAI_BASE_URL, no local weights)
EMBEDDING_OPTION = os.getenv("EMBEDDING_OPTION", "local")
# Model name DashScopeCloudEmbedder calls via dashscope.MultiModalEmbedding
DASHSCOPE_EMBEDDING_MODEL_NAME = os.getenv("DASHSCOPE_EMBEDDING_MODEL_NAME", "tongyi-embedding-vision-plus")

# Model Checkpoints (used if VLM_OPTION="local" or during local embeddings)
QWEN_VLM_MODEL_ID = os.getenv("QWEN_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_EMBEDDING_MODEL_ID = os.getenv("QWEN_EMBEDDING_MODEL_ID", "Qwen/Qwen2.5-VL-8B-Instruct")
M2D_CLAP_MODEL_ID = os.getenv("M2D_CLAP_MODEL_ID", "weights/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth")

# Zero-shot Object Detector Checkpoint (YOLOE-26, open-vocabulary text-prompt detection)
YOLOE_MODEL_ID = os.getenv("YOLOE_MODEL_ID", "yoloe-26x-seg.pt")
DETECTOR_OPTION = os.getenv("DETECTOR_OPTION", "local")
DETECTION_CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", 0.15))

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Search settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 20))
RRF_CONSTANT = int(os.getenv("RRF_CONSTANT", 60))
VQA_BOX_THRESHOLD = float(os.getenv("VQA_BOX_THRESHOLD", 0.3))
# Minimum VLM rerank score for a Type 1 (Textual-KIS) candidate to be kept in
# results, filtering out low-relevance frames that only made the initial
# candidate pool via a noisy sparse/BM25 or HyDE match
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", 0.2))

# AIC competition scoring rewards submitting up to 100 ranked answers per
# query (Final Score averages R@1/5/20/50/100 - the best score within each
# k-sized prefix) - submitting only 1 answer makes R@1=R@5=...=R@100, wasting
# the credit available at higher k. SUBMISSION_TOP_K widens the candidate
# pool/output list toward that; RERANK_TOP_K keeps the (expensive, VLM-based)
# Type 1/2 rerank pass scoped to just the head of that pool - the tail
# (RERANK_TOP_K..SUBMISSION_TOP_K) is appended in original retrieval-rank
# order rather than costing a VLM call per candidate just to fill out
# R@50/R@100 with more ranked options.
SUBMISSION_TOP_K = int(os.getenv("SUBMISSION_TOP_K", 100))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 20))




VLM_MIN_PIXELS = int(os.getenv("VLM_MIN_PIXELS", 256 * 28 * 28))   # ~256 token/ảnh (sàn)
VLM_MAX_PIXELS = int(os.getenv("VLM_MAX_PIXELS", 768 * 28 * 28))   # ~768 token/ảnh (trần)
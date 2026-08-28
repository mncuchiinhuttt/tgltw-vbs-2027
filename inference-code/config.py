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
# Bound reasoning/output work for live VBS requests.  The value is an upper
# bound, not a request to generate a long answer; lower it (for example to
# 2048) when an endpoint prioritizes latency over difficult-query accuracy.
OPENAI_VLM_MAX_COMPLETION_TOKENS = int(os.getenv("OPENAI_VLM_MAX_COMPLETION_TOKENS", 4096))
OPENAI_VLM_TIMEOUT_SEC = float(os.getenv("OPENAI_VLM_TIMEOUT_SEC", 45.0))
# How many OpenAIVLM.generate_batch() requests to issue concurrently. Raise
# this when OPENAI_BASE_URL points at a self-hosted vLLM server (see
# host_vllm.sh at the repo root) to get its continuous-batching throughput benefit.
VLM_BATCH_CONCURRENCY = int(os.getenv("VLM_BATCH_CONCURRENCY", 4))

# Model configuration options
# Options: "local" (uses Qwen3-VL locally) or "openai" (uses GPT 5.5 Pro / GPT-4o style API)
VLM_OPTION = os.getenv("VLM_OPTION", "openai")
# Options: "local" (QwenVL8BEmbedder loading QWEN_EMBEDDING_MODEL_ID below - ~4-5GB
# for the default 2B checkpoint, ~15GB if overridden to the 8B one) or "cloud"
# (DashScopeCloudEmbedder via OPENAI_API_KEY/OPENAI_BASE_URL, no local weights)
EMBEDDING_OPTION = os.getenv("EMBEDDING_OPTION", "local")
# Model name DashScopeCloudEmbedder calls via dashscope.MultiModalEmbedding
DASHSCOPE_EMBEDDING_MODEL_NAME = os.getenv("DASHSCOPE_EMBEDDING_MODEL_NAME", "tongyi-embedding-vision-plus")

# Model Checkpoints (used if VLM_OPTION="local" or during local embeddings)
QWEN_VLM_MODEL_ID = os.getenv("QWEN_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
# Default visual embedding model: Tencent WeMM-Embedding-4B (4B params multimodal embedder)
VISUAL_EMBEDDING_MODEL_ID = os.getenv("VISUAL_EMBEDDING_MODEL_ID", "tencent/WeMM-Embedding-4B")
QWEN_EMBEDDING_MODEL_ID = os.getenv("QWEN_EMBEDDING_MODEL_ID", VISUAL_EMBEDDING_MODEL_ID)
# Optional MRL truncation: truncate + re-normalize QwenVL8BEmbedder's output
# to this many leading dimensions (e.g. 512) for a smaller/faster Qdrant
# index at a small recall cost - a train-time property of the Qwen3-VL-Embedding
# checkpoints (arXiv:2601.04720), not something that needs retraining or a
# second model. Leave unset to keep the full embedding dimension (no-op,
# same behavior as before this option existed).
EMBEDDING_MRL_DIM = int(os.getenv("EMBEDDING_MRL_DIM")) if os.getenv("EMBEDDING_MRL_DIM") else None
M2D_CLAP_MODEL_ID = os.getenv("M2D_CLAP_MODEL_ID", "weights/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth")

# Zero-shot Object Detector Checkpoint (YOLOE-26, open-vocabulary text-prompt detection)
YOLOE_MODEL_ID = os.getenv("YOLOE_MODEL_ID", "yoloe-26x-seg.pt")
DETECTOR_OPTION = os.getenv("DETECTOR_OPTION", "local")
DETECTION_CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", 0.15))

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
VISUAL_COLLECTION_NAME = os.getenv("VISUAL_COLLECTION_NAME", os.getenv("QDRANT_COLLECTION_NAME", "visual_keyframes_v1"))
QDRANT_COLLECTION_NAME = VISUAL_COLLECTION_NAME
# Search settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 20))
RRF_CONSTANT = int(os.getenv("RRF_CONSTANT", 60))

# H-EAGLE-lite coarse-to-fine retrieval.  The shot collection is populated by
# preprocessing but the route stays off until a corpus benchmark confirms no
# recall regression.  Turning it on only narrows the dense frame branch;
# sparse text matching continues to provide a global OCR/metadata path.
HEAGLE_LITE_ENABLED = os.getenv("HEAGLE_LITE_ENABLED", "false").lower() == "true"
HEAGLE_SHOT_TOP_K = int(os.getenv("HEAGLE_SHOT_TOP_K", 64))
HEAGLE_FRAME_MULTIPLIER = max(1, int(os.getenv("HEAGLE_FRAME_MULTIPLIER", 4)))

# VBS is a LIVE/interactive competition, not AIC's batch Sơ tuyển round - an
# operator issues queries live and every second counts toward the task's
# KIS score (score = 50 + (300-t)/6 - 10*|wrong submissions|). Qdrant's
# default approximate HNSW search is the right choice here (fast, small
# recall trade-off), unlike AIC's batch/4h-window setting where brute-force
# exact search cost nothing against the time budget. Set to true only if
# testing shows HNSW's recall loss is unacceptable and per-query latency
# still fits comfortably within a task's 5-7 minute clock.
QDRANT_EXACT_SEARCH = os.getenv("QDRANT_EXACT_SEARCH", "false").lower() == "true"
VQA_BOX_THRESHOLD = float(os.getenv("VQA_BOX_THRESHOLD", 0.3))
# Minimum VLM rerank score for a Type 1 (Textual-KIS) candidate to be kept in
# results, filtering out low-relevance frames that only made the initial
# candidate pool via a noisy sparse/BM25 or HyDE match
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", 0.2))

# Inherited from AIC's batch scoring (rewarded submitting up to 100 ranked
# answers per query, R@1/5/20/50/100) - VBS's KIS submission is a single
# answer per attempt instead, so SUBMISSION_TOP_K here just controls how deep
# the operator's browsable result grid goes, not a submission format.
# RERANK_TOP_K still keeps the (expensive) VLM rerank pass scoped to the head
# of that pool - the tail is appended in original retrieval-rank order.
SUBMISSION_TOP_K = int(os.getenv("SUBMISSION_TOP_K", 100))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 20))

# Verification Reranking (Fusionista2.0-inspired, MMM 2026 LNCS 16415 ch.17
# "Reranking with Interactive Confirmation"): an LLM breaks the query into a
# few yes/no checks on specific attributes/objects/actions, and each
# candidate is verified against them - catching cases where a candidate
# superficially matches the query's embedding/VLM-similarity score but fails
# a specific attribute check. Adds VERIFICATION_NUM_QUESTIONS extra VLM calls
# per reranked candidate (bounded by RERANK_TOP_K above) EVERY search - fine
# for AIC's 4-hour batch window, too slow as a default for VBS's live 5-7
# minute-per-task clock. Leave off by default; an operator can still flip
# this on mid-task via env/restart if a task has time to spare, or it can be
# wired to a manual "verify these results" UI action later.
VERIFICATION_RERANK_ENABLED = os.getenv("VERIFICATION_RERANK_ENABLED", "false").lower() == "true"
VERIFICATION_NUM_QUESTIONS = int(os.getenv("VERIFICATION_NUM_QUESTIONS", 3))
# Type 1 blend: (1 - VERIFICATION_WEIGHT_TYPE1) * vlm_score + VERIFICATION_WEIGHT_TYPE1 * verification_ratio
VERIFICATION_WEIGHT_TYPE1 = float(os.getenv("VERIFICATION_WEIGHT_TYPE1", 0.3))
# Type 2 blend replaces the old fixed 0.4 rrf / 0.6 vqa split with a 3-way
# weighted sum (should sum to 1.0)
TYPE2_RRF_WEIGHT = float(os.getenv("TYPE2_RRF_WEIGHT", 0.3))
TYPE2_VQA_WEIGHT = float(os.getenv("TYPE2_VQA_WEIGHT", 0.5))
TYPE2_VERIFICATION_WEIGHT = float(os.getenv("TYPE2_VERIFICATION_WEIGHT", 0.2))

# TRAKE (Type 3): how many top candidate videos get a full DP alignment pass
# (Reranker.rerank_type3_temporal) - each pass costs one Qdrant scroll +
# len(events) text embedding calls, so this is capped well below
# SUBMISSION_TOP_K rather than aligning every distinct video in the pool.
TRAKE_MAX_VIDEOS_TO_ALIGN = int(os.getenv("TRAKE_MAX_VIDEOS_TO_ALIGN", 20))




# Secondary embedder ensemble (Fusionista2.0/VERGE-inspired, VBS2026 - see
# models/siglip_embedder.py + preprocessing/config.py for the indexing side).
# Must match whatever preprocessing was actually run with - if the
# "visual_index" collection wasn't (re)built with a named "siglip" vector,
# leave this false regardless of preprocessing/config.py's setting.
SECONDARY_EMBEDDER_ENABLED = os.getenv("SECONDARY_EMBEDDER_ENABLED", "false").lower() == "true"
SIGLIP_MODEL_ID = os.getenv("SIGLIP_MODEL_ID", "google/siglip-so400m-patch14-384")

VLM_MIN_PIXELS = int(os.getenv("VLM_MIN_PIXELS", 256 * 28 * 28))   # ~256 token/ảnh (sàn)
VLM_MAX_PIXELS = int(os.getenv("VLM_MAX_PIXELS", 768 * 28 * 28))   # ~768 token/ảnh (trần)
# Lower-cost secondary images used by the shared OpenAI/Qwen multi-image
# pathway. Keep these in inference config as well as preprocessing config so
# importing either VLM backend is self-contained.
FAST_PATHWAY_MIN_PIXELS = int(os.getenv("FAST_PATHWAY_MIN_PIXELS", 64 * 28 * 28))
FAST_PATHWAY_MAX_PIXELS = int(os.getenv("FAST_PATHWAY_MAX_PIXELS", 128 * 28 * 28))

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

# AIC's Sơ tuyển round is NOT a live/interactive setting like VBS (BTC sends
# ~45 queries once, we have 4 hours to submit all of them) - so there's no
# per-query latency pressure that would justify Qdrant's default approximate
# HNSW search. QDRANT_EXACT_SEARCH runs a full brute-force scan instead
# (U-Cker/VBS2026, arXiv LNCS 16415 ch.18 - "we prioritize exact computation
# in order to guarantee reliability... rather than approximate nearest
# neighbor methods"), trading query latency (still well within the 4h
# budget at our dataset scale) for zero HNSW recall loss. Set to false to
# fall back to approximate search if the collection ever grows large enough
# that even brute-force scan blows the time budget.
QDRANT_EXACT_SEARCH = os.getenv("QDRANT_EXACT_SEARCH", "true").lower() == "true"
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

# Verification Reranking (Fusionista2.0-inspired, MMM 2026 LNCS 16415 ch.17
# "Reranking with Interactive Confirmation"): an LLM breaks the query into a
# few yes/no checks on specific attributes/objects/actions, and each
# candidate is verified against them - catching cases where a candidate
# superficially matches the query's embedding/VLM-similarity score but fails
# a specific attribute check. Adds VERIFICATION_NUM_QUESTIONS extra VLM calls
# per reranked candidate (bounded by RERANK_TOP_K above), which is affordable
# given the Sơ tuyển round's 4-hour batch submission window rather than
# VBS-style live per-query latency.
VERIFICATION_RERANK_ENABLED = os.getenv("VERIFICATION_RERANK_ENABLED", "true").lower() == "true"
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
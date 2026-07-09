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

# Model configuration options
# Options: "local" (uses Qwen3-VL locally) or "openai" (uses GPT 5.5 Pro / GPT-4o style API)
VLM_OPTION = os.getenv("VLM_OPTION", "openai")
# Options: "local" (QwenVL8BEmbedder, ~15GB) or "cloud" (DashScopeCloudEmbedder,
# tongyi-embedding-vision-plus via OPENAI_API_KEY/OPENAI_BASE_URL, no local weights)
EMBEDDING_OPTION = os.getenv("EMBEDDING_OPTION", "local")

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

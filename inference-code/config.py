import os
from dotenv import load_dotenv

# Load env variables from .env file if it exists
load_dotenv()

# API Keys & Endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DINO_X_API_KEY = os.getenv("DINO_X_API_KEY", "")
DINO_X_API_URL = os.getenv("DINO_X_API_URL", "https://api.dino-x.ai/v1/detect")

# Model configuration options
# Options: "local" (uses Qwen3-VL locally) or "openai" (uses GPT 5.5 Pro / GPT-4o style API)
VLM_OPTION = os.getenv("VLM_OPTION", "openai")

# Options: 
# - "dino-x" (uses online API)
# - "dino-x-local" (uses local self-hosted weights offline)
# - "grounding-dino" (uses offline local Grounding DINO 1.5 Pro)
DETECTOR_OPTION = os.getenv("DETECTOR_OPTION", "dino-x-local")

# Model Checkpoints (used if VLM_OPTION="local" or during local embeddings)
QWEN_VLM_MODEL_ID = os.getenv("QWEN_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_EMBEDDING_MODEL_ID = os.getenv("QWEN_EMBEDDING_MODEL_ID", "Qwen/Qwen2.5-VL-8B-Instruct")
M2D_CLAP_MODEL_ID = os.getenv("M2D_CLAP_MODEL_ID", "laion/clap-htsat-fused")

# Local self-hosted DINO-X weights & config paths
DINO_X_LOCAL_MODEL_PATH = os.getenv("DINO_X_LOCAL_MODEL_PATH", "weights/dino-x-pro.pth")
DINO_X_LOCAL_CONFIG_PATH = os.getenv("DINO_X_LOCAL_CONFIG_PATH", "configs/dino-x-pro.py")

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Search settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 20))
RRF_CONSTANT = int(os.getenv("RRF_CONSTANT", 60))
VQA_BOX_THRESHOLD = float(os.getenv("VQA_BOX_THRESHOLD", 0.3))

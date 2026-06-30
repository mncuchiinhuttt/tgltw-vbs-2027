import os
from dotenv import load_dotenv

# Load env variables from .env file if it exists
load_dotenv()

# API Keys & Endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DINO_X_API_KEY = os.getenv("DINO_X_API_KEY", "")
DINO_X_API_URL = os.getenv("DINO_X_API_URL", "https://api.dino-x.ai/v1/detect")

# Model configuration options
# Options: "local" (uses Qwen3-VL via Hugging Face) or "openai" (uses GPT 5.5 Pro / GPT-4o style API)
VLM_OPTION = os.getenv("VLM_OPTION", "openai")

# Options: 
# - "dino-x" (uses online API)
# - "dino-x-local" (uses self-hosted/offline DINO-X model local weights)
# - "grounding-dino" (uses offline local Grounding DINO 1.5 Pro)
DETECTOR_OPTION = os.getenv("DETECTOR_OPTION", "dino-x")

# Model Checkpoints (used if VLM_OPTION="local" or during local embeddings/transcription)
QWEN_VLM_MODEL_ID = os.getenv("QWEN_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_EMBEDDING_MODEL_ID = os.getenv("QWEN_EMBEDDING_MODEL_ID", "Qwen/Qwen2.5-VL-8B-Instruct")
PHOWHISPER_MODEL_ID = os.getenv("PHOWHISPER_MODEL_ID", "vinai/PhoWhisper-large")
M2D_CLAP_MODEL_ID = os.getenv("M2D_CLAP_MODEL_ID", "laion/clap-htsat-fused")

# Self-hosted / Local DINO-X config
DINO_X_LOCAL_MODEL_PATH = os.getenv("DINO_X_LOCAL_MODEL_PATH", "weights/dino-x-pro.pth")
DINO_X_LOCAL_CONFIG_PATH = os.getenv("DINO_X_LOCAL_CONFIG_PATH", "configs/dino-x-pro.py")

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Pipeline Parameters
SCENE_DETECTION_THRESHOLD = float(os.getenv("SCENE_DETECTION_THRESHOLD", 27.0))
KEYFRAME_DIVERSITY_THRESHOLD = float(os.getenv("KEYFRAME_DIVERSITY_THRESHOLD", 0.15))

# Object Detection classes to search for in video frames
OBJECT_DETECTION_PROMPTS = ["xe máy", "biển số xe", "bảng hiệu", "cờ", "người", "ô tô", "xe đạp"]

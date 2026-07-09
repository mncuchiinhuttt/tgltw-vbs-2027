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
# Options: "local" (uses Qwen3-VL via Hugging Face) or "openai" (uses GPT 5.5 Pro / GPT-4o style API)
VLM_OPTION = os.getenv("VLM_OPTION", "openai")
# Options: "local" (QwenVL8BEmbedder, ~15GB) or "cloud" (DashScopeCloudEmbedder,
# tongyi-embedding-vision-plus via OPENAI_API_KEY/OPENAI_BASE_URL, no local weights)
EMBEDDING_OPTION = os.getenv("EMBEDDING_OPTION", "local")

# Model Checkpoints (used if VLM_OPTION="local" or during local embeddings/transcription)
QWEN_VLM_MODEL_ID = os.getenv("QWEN_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_EMBEDDING_MODEL_ID = os.getenv("QWEN_EMBEDDING_MODEL_ID", "Qwen/Qwen2.5-VL-8B-Instruct")
PHOWHISPER_MODEL_ID = os.getenv("PHOWHISPER_MODEL_ID", "vinai/PhoWhisper-large")
M2D_CLAP_MODEL_ID = os.getenv("M2D_CLAP_MODEL_ID", "weights/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth")

# Zero-shot Object Detector Checkpoint (YOLOE-26, open-vocabulary text-prompt detection)
# Scales: yoloe-26{n,s,m,l,x}-seg.pt - x is the most accurate/slowest
YOLOE_MODEL_ID = os.getenv("YOLOE_MODEL_ID", "yoloe-26x-seg.pt")
DETECTOR_OPTION = os.getenv("DETECTOR_OPTION", "local")
DETECTION_CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", 0.15))

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Pipeline Parameters
SCENE_DETECTION_THRESHOLD = float(os.getenv("SCENE_DETECTION_THRESHOLD", 27.0))
KEYFRAME_DIVERSITY_THRESHOLD = float(os.getenv("KEYFRAME_DIVERSITY_THRESHOLD", 0.15))

# Object Detection classes to search for in video frames. Labels are Vietnamese
# (stored/reported/indexed for BM25 matching against Vietnamese queries); the
# parallel English list is only used to compute CLIP text embeddings for YOLOE,
# since its text encoder is primarily English-trained and matches noticeably
# better on English phrasing than on the Vietnamese equivalent.
OBJECT_DETECTION_PROMPTS = ["xe máy", "biển số xe", "bảng hiệu", "cờ", "người", "ô tô", "xe đạp"]
OBJECT_DETECTION_PROMPTS_EN = ["motorbike", "license plate", "signboard", "flag", "person", "car", "bicycle"]

# Subset of the above categories that are small enough to get lost when a
# full frame is downscaled to the detector's input size (e.g. license plates
# are tiny in dashcam-style footage). These get a supplementary tiled pass
# (see ObjectDetector.detect_tiled) merged into the full-frame detections.
TILED_DETECTION_LABELS = ["biển số xe"]
TILED_DETECTION_LABELS_EN = ["license plate"]

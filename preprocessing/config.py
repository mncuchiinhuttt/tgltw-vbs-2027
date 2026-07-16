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
# How many OpenAIVLM.generate_batch() requests to issue concurrently. Only
# matters for batch/concurrent-serving backends (self-hosted vLLM, or a
# provider that handles concurrent requests efficiently) - raise this when
# OPENAI_BASE_URL points at a self-hosted vLLM server (see host_vllm.sh) to
# actually get its continuous-batching throughput benefit.
VLM_BATCH_CONCURRENCY = int(os.getenv("VLM_BATCH_CONCURRENCY", 4))

# Model configuration options
# Options: "local" (uses Qwen3-VL via Hugging Face) or "openai" (uses GPT 5.5 Pro / GPT-4o style API)
VLM_OPTION = os.getenv("VLM_OPTION", "openai")
# Options: "local" (QwenVL8BEmbedder, ~15GB) or "cloud" (DashScopeCloudEmbedder,
# via OPENAI_API_KEY/OPENAI_BASE_URL, no local weights)
EMBEDDING_OPTION = os.getenv("EMBEDDING_OPTION", "local")
# Model name DashScopeCloudEmbedder calls via dashscope.MultiModalEmbedding
DASHSCOPE_EMBEDDING_MODEL_NAME = os.getenv("DASHSCOPE_EMBEDDING_MODEL_NAME", "tongyi-embedding-vision-plus")

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

# Adaptive Keyframe Sampling: a lightweight CLIP pass estimates how "static"
# vs "dynamic" a scene is (variance of per-frame embeddings across the
# scene's candidates), which sets a per-scene keyframe budget - static scenes
# (e.g. talking heads) keep few frames, dynamic scenes keep more, capped at
# KEYFRAME_MAX_BUDGET. The actual frame selection within that budget still
# uses farthest-point sampling over the full Qwen3-Embedding-VL-8B space.
KEYFRAME_VARIANCE_LOW = float(os.getenv("KEYFRAME_VARIANCE_LOW", 0.01))
KEYFRAME_VARIANCE_MID = float(os.getenv("KEYFRAME_VARIANCE_MID", 0.05))
KEYFRAME_MAX_BUDGET = int(os.getenv("KEYFRAME_MAX_BUDGET", 8))

# OCR settings (PP-OCRv6 detection+recognition, replacing the previous
# VLM-based OCR path - the VLM is now only used to re-read individual crops
# PP-OCRv6 recognized with low confidence)
OCR_LANG = os.getenv("OCR_LANG", "vi")
OCR_REC_SCORE_THRESHOLD = float(os.getenv("OCR_REC_SCORE_THRESHOLD", 0.5))
# Supplementary overlapping-tile OCR pass (mirrors ObjectDetector.detect_tiled)
# to catch small/corner text a full-frame pass might downscale away. Off by
# default: it multiplies OCR cost per keyframe by roughly the tile count, and
# PP-OCRv6's own full-frame detection already handles most cases well.
OCR_USE_TILING = os.getenv("OCR_USE_TILING", "false").lower() == "true"

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

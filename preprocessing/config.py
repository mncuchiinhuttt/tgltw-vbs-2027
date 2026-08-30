import os
from pathlib import Path
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
# Shared OpenAI-compatible VLM runtime bounds. Keep preprocessing and live
# inference on the same request contract when they import models/openai_vlm.
OPENAI_VLM_MAX_COMPLETION_TOKENS = int(os.getenv("OPENAI_VLM_MAX_COMPLETION_TOKENS", 4096))
OPENAI_VLM_TIMEOUT_SEC = float(os.getenv("OPENAI_VLM_TIMEOUT_SEC", 45.0))
# How many OpenAIVLM.generate_batch() requests to issue concurrently. Only
# matters for batch/concurrent-serving backends (self-hosted vLLM, or a
# provider that handles concurrent requests efficiently) - raise this when
# OPENAI_BASE_URL points at a self-hosted vLLM server (see host_vllm.sh) to
# actually get its continuous-batching throughput benefit.
VLM_BATCH_CONCURRENCY = int(os.getenv("VLM_BATCH_CONCURRENCY", 4))

# Model configuration options
# VLM is independent; the visual/text embedding is fixed to WeMM-Embedding-4B.
VLM_OPTION = os.getenv("VLM_OPTION", "openai")
EMBEDDING_OPTION = "local"
DASHSCOPE_EMBEDDING_MODEL_NAME = os.getenv("DASHSCOPE_EMBEDDING_MODEL_NAME", "tongyi-embedding-vision-plus")

# Model checkpoints
QWEN_VLM_MODEL_ID = os.getenv("QWEN_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
VISUAL_EMBEDDING_MODEL_ID = "tencent/WeMM-Embedding-4B"
QWEN_EMBEDDING_MODEL_ID = VISUAL_EMBEDDING_MODEL_ID
EMBEDDING_MRL_DIM = int(os.getenv("EMBEDDING_MRL_DIM")) if os.getenv("EMBEDDING_MRL_DIM") else None
# ASR (faster-whisper / CTranslate2). Whisper large-v3-turbo: 99-language,
# MIT. Must be a CT2-converted repo id (download_assets.py runs `hf download`
# on it, so faster-whisper's "turbo" shorthand is not usable here).
ASR_MODEL_ID = os.getenv("ASR_MODEL_ID", "deepdml/faster-whisper-large-v3-turbo-ct2")
# Blank = auto-detect per file (V3C is multilingual). Set e.g. "en" to force.
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "") or None
# CTranslate2 compute type: "auto" picks float16 on CUDA / int8 on CPU.
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "auto")
ASR_VAD_FILTER_ENABLED = os.getenv("ASR_VAD_FILTER_ENABLED", "true").lower() == "true"
ASR_WORD_TIMESTAMPS_ENABLED = os.getenv("ASR_WORD_TIMESTAMPS_ENABLED", "true").lower() == "true"
# Hallucination/silence filter thresholds - OpenAI Whisper's own reference
# values. faster-whisper reports these per segment but does NOT drop the
# segment for us, so we filter before embedding+indexing.
ASR_MIN_AVG_LOGPROB = float(os.getenv("ASR_MIN_AVG_LOGPROB", -1.0))
ASR_MAX_NO_SPEECH_PROB = float(os.getenv("ASR_MAX_NO_SPEECH_PROB", 0.6))
ASR_MAX_COMPRESSION_RATIO = float(os.getenv("ASR_MAX_COMPRESSION_RATIO", 2.4))
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
QDRANT_ALLOW_RECREATE = os.getenv("QDRANT_ALLOW_RECREATE", "false").lower() == "true"
# Set this only when intentionally rebuilding one video's old points after a
# detector/model change.  It is off by default because deleting a video before
# a crash would leave that video temporarily unindexed.
QDRANT_REBUILD_VIDEO_ON_START = os.getenv("QDRANT_REBUILD_VIDEO_ON_START", "false").lower() == "true"
# Uploading one point per HTTP request is needlessly expensive for a large
# offline index.  The indexer buffers both visual and ambient-audio points and
# flushes this many at a time; the final flush is explicit in main.py.
QDRANT_UPSERT_BATCH_SIZE = max(1, int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", 128)))

# Pipeline Parameters
SCENE_DETECTION_THRESHOLD = float(os.getenv("SCENE_DETECTION_THRESHOLD", 27.0))

# Shot boundary detection.  Official V3C shot maps always take precedence in
# main.py.  For raw videos, TransNetV2 is the default and PySceneDetect stays
# as a deliberate runtime fallback until the detector has been benchmarked on
# the mounted corpus.
SHOT_DETECTOR = os.getenv("SHOT_DETECTOR", "transnetv2").lower()
SHOT_DETECTOR_FALLBACK = os.getenv("SHOT_DETECTOR_FALLBACK", "pyscenedetect").lower()
SHOT_BENCHMARK_MODE = os.getenv("SHOT_BENCHMARK_MODE", "false").lower() == "true"
TRANSNETV2_MODEL_PATH = os.getenv(
    "TRANSNETV2_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "weights" / "transnetv2-pytorch-weights.pth"),
)
TRANSNETV2_DEVICE = os.getenv("TRANSNETV2_DEVICE", "auto")
TRANSNETV2_BATCH_SIZE = max(1, int(os.getenv("TRANSNETV2_BATCH_SIZE", 1)))
TRANSNETV2_THRESHOLD = float(os.getenv("TRANSNETV2_THRESHOLD", 0.5))

# Adaptive Keyframe Sampling: a lightweight CLIP pass estimates how "static"
# vs "dynamic" a scene is (variance of per-frame embeddings across the
# scene's candidates), which sets a per-scene keyframe budget - static scenes
# (e.g. talking heads) keep few frames, dynamic scenes keep more, capped at
# KEYFRAME_MAX_BUDGET. The actual frame selection within that budget still
# uses farthest-point sampling over the full Tencent WeMM-Embedding-4B space.
KEYFRAME_VARIANCE_LOW = float(os.getenv("KEYFRAME_VARIANCE_LOW", 0.01))
KEYFRAME_VARIANCE_MID = float(os.getenv("KEYFRAME_VARIANCE_MID", 0.05))
KEYFRAME_MAX_BUDGET = int(os.getenv("KEYFRAME_MAX_BUDGET", 8))
# Cheap CPU-only quality bonus used by farthest-point sampling.  It favors
# sharp, readable frames without replacing semantic diversity or adding a
# learned saliency model.  Set to 0 to reproduce the old selector exactly.
KEYFRAME_SHARPNESS_WEIGHT = max(0.0, float(os.getenv("KEYFRAME_SHARPNESS_WEIGHT", 0.15)))

# DAKE (Dynamic-Aware Keyframe Extraction, U-CESE arXiv:2605.23274) - a
# training-free pre-filter run before Adaptive Keyframe Sampling's CLIP pass
# above: JPEG-encodes each candidate frame and measures the rate of change
# in compressed file size between nearby frames as a free motion proxy (no
# model inference at all), keeping only the highest-"steepness" fraction of
# candidates. Supplementary, not a replacement - AKS's own CLIP-variance
# budget and Qwen-embedding farthest-point sampling still run afterward on
# whatever this keeps; it only cuts how many candidates those (much more
# expensive) passes have to process.
KEYFRAME_DAKE_ENABLED = os.getenv("KEYFRAME_DAKE_ENABLED", "true").lower() == "true"
KEYFRAME_DAKE_RATIO = float(os.getenv("KEYFRAME_DAKE_RATIO", 0.5))
KEYFRAME_DAKE_WINDOW = int(os.getenv("KEYFRAME_DAKE_WINDOW", 3))
# DAKE's own safeguard ("at least one keyframe every 2x fps frames") adapted
# to our already-~1fps-downsampled candidate stream (extract_candidate_frames'
# default sampling_rate_fps) - expressed directly as a candidate-list index
# gap rather than a native-video-fps multiple.
KEYFRAME_DAKE_MAX_GAP = int(os.getenv("KEYFRAME_DAKE_MAX_GAP", 4))

# SSM-inspired scene boundary refinement (VIREO at VBS2026, MMM 2026 LNCS
# 16415 ch.19 - "Merging weak boundaries"): PySceneDetect's ContentDetector
# uses one fixed threshold for the whole video, which VIREO found
# under-segments dense/short videos and over-segments long uniform ones.
# Rather than reimplementing their full self-similarity-matrix + kernel
# temporal segmentation pipeline (built on BLIP embeddings), this scopes
# down to just their "local refinement" + "merging weak boundaries" steps
# as an optional post-process on PySceneDetect's own candidate cuts, reusing
# the already-loaded LightweightCLIPEmbedder instead of adding a second
# heavy model (BLIP) just for this. Preprocessing-only cost (offline, not
# against the Sơ tuyển round's 4-hour submission window) - OFF by default
# since it's the least-certain-payoff of this round's changes for a
# V3C-like general dataset (VIREO's own gain was specific to MVK/LHE75's
# unusually short/long, homogeneous videos).
SCENE_MERGE_ENABLED = os.getenv("SCENE_MERGE_ENABLED", "false").lower() == "true"
SCENE_MERGE_WINDOW_SEC = float(os.getenv("SCENE_MERGE_WINDOW_SEC", 1.5))
SCENE_MERGE_SAMPLES_PER_SIDE = int(os.getenv("SCENE_MERGE_SAMPLES_PER_SIDE", 3))
# A boundary is "weak" (merged away) when cross-boundary similarity is at
# least this fraction of the average within-scene similarity of its two
# neighbors - i.e. the cut barely separates anything relative to how much
# each side already varies internally.
SCENE_MERGE_THRESHOLD_RATIO = float(os.getenv("SCENE_MERGE_THRESHOLD_RATIO", 0.9))

# Optional V3C/VBS official assets.  When V3C_ASSETS_DIR is empty, main.py
# probes --data_dir itself for msb/, keyframes/, metadata/, and asr/.  Every
# asset family is independently optional and falls back to local processing.
V3C_ASSETS_ENABLED = os.getenv("V3C_ASSETS_ENABLED", "true").lower() == "true"
V3C_ASSETS_DIR = os.getenv("V3C_ASSETS_DIR", "")
V3C_OFFICIAL_KEYFRAMES_ENABLED = os.getenv("V3C_OFFICIAL_KEYFRAMES_ENABLED", "true").lower() == "true"

# OCR settings (PP-OCRv6 detection + recognition, gated by SAM3
# region proposal - see SAM3/SAHI settings below)
OCR_LANG = os.getenv("OCR_LANG", "en")
# If PP-OCRv6's recognition confidence is below this threshold, the
# lightweight fallback VLM re-reads the crop.
OCR_REC_SCORE_THRESHOLD = float(os.getenv("OCR_REC_SCORE_THRESHOLD", 0.5))

# Object Detection classes to search for in video frames. Labels are Vietnamese
# (stored/reported/indexed for BM25 matching against Vietnamese queries); the
# parallel English list is only used to compute CLIP text embeddings for YOLOE,
# since its text encoder is primarily English-trained and matches noticeably
# better on English phrasing than on the Vietnamese equivalent.
OBJECT_DETECTION_PROMPTS = ["xe máy", "biển số xe", "bảng hiệu", "cờ", "người", "ô tô", "xe đạp"]
OBJECT_DETECTION_PROMPTS_EN = ["motorbike", "license plate", "signboard", "flag", "person", "car", "bicycle"]

# SAM3 region-proposal pre-filter (Promptable Concept Segmentation, ~0.9B,
# zero-shot, gated on Hugging Face - accept the license at
# https://huggingface.co/facebook/sam3 and set HF_TOKEN before first run).
# Gates BOTH Object Detection and OCR: SAHI-style tiling + downstream
# detection/recognition only run inside the candidate boxes SAM3 proposes for
# a keyframe, and the whole detection/OCR step is skipped for that keyframe
# if SAM3 finds no matching region at all.
SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
REGION_PROPOSAL_CONF_THRESHOLD = float(os.getenv("REGION_PROPOSAL_CONF_THRESHOLD", 0.3))

# SAHI-style tiling parameters applied only within SAM3's candidate regions
# (hand-rolled tiling reused from ObjectDetector.detect_tiled /
# TextDetectorOCR._detect_recognize_tiled - SAHI's own model registry has no
# PaddleOCR backend to plug into, see the comment in preprocessing/video/ocr.py).
SAHI_TILE_SIZE = int(os.getenv("SAHI_TILE_SIZE", 512))
SAHI_TILE_OVERLAP = float(os.getenv("SAHI_TILE_OVERLAP", 0.2))

# General concept prompts SAM3 proposes candidate regions for, ahead of the
# specific-class YOLOE pass run per region.
OBJECT_REGION_CONCEPTS_EN = ["text or sign region", "human", "vehicle", "small distinct object"]
# Concept prompt SAM3 proposes text/sign regions for, ahead of the OCR pass.
OCR_REGION_CONCEPTS_EN = ["text or sign region"]

# Conditional Super-Resolution (Real-ESRGAN x4): OCR crops shorter than this
# many pixels get upscaled before recognition; taller crops skip SR and go
# straight to recognition.
OCR_SR_MIN_HEIGHT_PX = int(os.getenv("OCR_SR_MIN_HEIGHT_PX", 16))
REAL_ESRGAN_MODEL_ID = os.getenv("REAL_ESRGAN_MODEL_ID", "RealESRGAN_x4plus.pth")

# Dedicated lightweight fallback VLM for crops where the recognition
# confidence is still below OCR_REC_SCORE_THRESHOLD -
# deliberately NOT the main captioning VLM (QwenVLM/OpenAIVLM), which is far
# more expensive to run per-crop.
FALLBACK_VLM_MODEL_ID = os.getenv("FALLBACK_VLM_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")


# Secondary embedder ensemble (Fusionista2.0/VERGE-inspired, VBS2026 - see
# models/siglip_embedder.py). Opt-in and OFF by default: enabling it means
# preprocessing must compute a second embedding per keyframe and the Qdrant
# "visual_index" collection must be (re)created with a named "siglip" vector
# alongside the primary one - a one-time reprocessing cost paid before the
# Sơ tuyển round's query day, not against its 4-hour submission window, but
# real GPU/wall-clock time nonetheless. Existing single-vector collections
# are unaffected until this is turned on and preprocessing is re-run.
SECONDARY_EMBEDDER_ENABLED = os.getenv("SECONDARY_EMBEDDER_ENABLED", "false").lower() == "true"
SIGLIP_MODEL_ID = os.getenv("SIGLIP_MODEL_ID", "google/siglip-so400m-patch14-384")

VLM_MIN_PIXELS = int(os.getenv("VLM_MIN_PIXELS", 256 * 28 * 28))   # ~256 token/ảnh (sàn)
VLM_MAX_PIXELS = int(os.getenv("VLM_MAX_PIXELS", 768 * 28 * 28))   # ~768 token/ảnh (trần)

# Slow/Fast dual-budget pathway (Method 2) - Fast pathway adds MORE frames per
# scene for motion coverage, so its per-frame pixel budget must sit well below
# VLM_MIN_PIXELS or total tokens per scene call go UP instead of down.
# fps params tuned for long/dense videos (higher motion-sampling density).
FAST_PATHWAY_DENSE_SAMPLING_FPS = float(os.getenv("FAST_PATHWAY_DENSE_SAMPLING_FPS", 8.0))  # decode fps cho optical flow
FAST_PATHWAY_FPS_TARGET = float(os.getenv("FAST_PATHWAY_FPS_TARGET", 2.5))                   # mật độ frame Fast mong muốn
FAST_PATHWAY_MIN_FRAMES = int(os.getenv("FAST_PATHWAY_MIN_FRAMES", 4))
FAST_PATHWAY_MAX_FRAMES = int(os.getenv("FAST_PATHWAY_MAX_FRAMES", 16))
FAST_PATHWAY_MIN_PIXELS = int(os.getenv("FAST_PATHWAY_MIN_PIXELS", 64 * 28 * 28))   # ~64 token/frame (sàn)
FAST_PATHWAY_MAX_PIXELS = int(os.getenv("FAST_PATHWAY_MAX_PIXELS", 128 * 28 * 28))  # ~128 token/frame (trần)

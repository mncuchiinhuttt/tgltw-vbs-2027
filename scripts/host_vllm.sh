#!/bin/bash
# Shell script to self-host the local VLM (e.g. Qwen3-VL) via vLLM's
# OpenAI-compatible server, for efficient batched/concurrent inference in
# both preprocessing/ and inference-code/, instead of one-request-at-a-time
# HuggingFace transformers calls (see models/qwen_vlm.py). Shared at the
# repo root since both modules point at it via the same models/openai_vlm.py
# client, the same way models/ itself is shared rather than duplicated.
#
# REQUIRES an NVIDIA (CUDA) or AMD (ROCm) GPU - vLLM does not support Apple
# Silicon/macOS or CPU-only inference at any usable speed. Run this on the
# actual GPU machine (a cloud instance, or the competition server), not a
# dev Mac.
#
# Once the server is up, point the pipeline at it via the existing
# OpenAI-compatible VLM client (models/openai_vlm.py) instead of loading a
# local HF model or calling a hosted API:
#   VLM_OPTION=openai
#   OPENAI_BASE_URL=http://localhost:8000/v1
#   OPENAI_API_KEY=not-needed        # vLLM's server doesn't check this by default
#   OPENAI_VLM_MODEL_NAME=<same $VLLM_MODEL given below>
#   VLM_BATCH_CONCURRENCY=16          # raise this to actually use vLLM's continuous batching

set -e

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"

echo "=== Checking host environment for vLLM self-hosting ==="

if ! command -v nvidia-smi &> /dev/null && ! command -v rocm-smi &> /dev/null; then
    echo "[ERROR] No NVIDIA (nvidia-smi) or AMD (rocm-smi) GPU detected on this host."
    echo "        vLLM requires CUDA or ROCm and does not run on Apple Silicon/CPU."
    echo "        Run this script on a GPU machine instead."
    exit 1
fi

if ! command -v vllm &> /dev/null; then
    echo "[ERROR] vllm is not installed. Install it with: pip install vllm"
    echo "        (see https://docs.vllm.ai/en/latest/getting_started/installation/)"
    exit 1
fi

echo "Model: $MODEL"
echo "Port: $PORT"
echo "Once ready, set OPENAI_BASE_URL=http://localhost:$PORT/v1 and OPENAI_VLM_MODEL_NAME=$MODEL"
echo "=== Starting vLLM OpenAI-compatible server ==="

vllm serve "$MODEL" \
    --port "$PORT" \
    --trust-remote-code

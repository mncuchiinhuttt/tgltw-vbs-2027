#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/riftuser/tgltw-vbs-2027"
cd "$REPO_DIR"

LOG_FILE="evaluation/indexing_logs/rag_benchmark_200.log"
mkdir -p "evaluation/indexing_logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 200-Query Benchmark Suite..." > "$LOG_FILE"

PYTHONPATH=inference-code:. uv run python evaluation/run_rag_benchmark.py \
  --query_file queries/vbs_rag_benchmark_200.json \
  --output_file evaluation/vbs_rag_benchmark_200_results.json \
  >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 200-Query Benchmark Completed!" >> "$LOG_FILE"

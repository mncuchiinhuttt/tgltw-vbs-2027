#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "=== Syncing the shared uv environment (preprocessing group) ==="
cd "${ROOT_DIR}"
uv sync --group preprocessing

echo "=== Setup complete ==="
echo "Run preprocessing with: uv run --group preprocessing python preprocessing/main.py --data_dir datasets"

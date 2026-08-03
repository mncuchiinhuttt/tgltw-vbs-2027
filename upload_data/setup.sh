#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "=== Syncing the shared uv environment (upload group) ==="
cd "${ROOT_DIR}"
uv sync --group upload

echo
echo "Environment ready. Run the uploader with:"
echo "uv run --group upload python upload_data/uploader.py"

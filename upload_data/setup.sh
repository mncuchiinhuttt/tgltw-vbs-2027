#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} not found on PATH." >&2
  exit 1
fi

echo "Creating virtual environment at ${VENV_DIR}..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "Upgrading pip..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip

echo "Installing Python packages..."
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"

echo
echo "Environment ready. Activate it with:"
echo "source \"${VENV_DIR}/bin/activate\""
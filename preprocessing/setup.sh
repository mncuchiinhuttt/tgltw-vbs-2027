#!/bin/bash
# Setup script for AIC2026 Preprocessing Pipeline

echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Setup complete ==="
echo "To activate environment run: source venv/bin/activate"

#!/bin/bash

# Thin wrapper around the Python launcher so both entry points share the same
# dependency bootstrap and process management.
set -e

cd "$(dirname "$0")"
exec python3 run_webapp.py

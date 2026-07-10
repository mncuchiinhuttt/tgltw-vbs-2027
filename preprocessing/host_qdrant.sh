#!/bin/bash
# Shell script to start Qdrant self-hosted server
# It will use Docker if available, otherwise it automatically downloads
# the precompiled standalone Qdrant binary for the local OS/Arch and runs it.

QDRANT_VERSION="v1.10.1"
PORT=6333
GRPC_PORT=6334

echo "=== Checking host environment for Qdrant self-hosting ==="

if command -v docker-compose &> /dev/null || (command -v docker &> /dev/null && docker compose version &> /dev/null); then
    echo "[INFO] Docker Compose detected. Starting Qdrant via Docker..."
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    else
        docker compose up -d
    fi
    echo "=== Qdrant started successfully via Docker ==="
    echo "Access Dashboard at: http://localhost:$PORT/dashboard"
    exit 0
elif command -v docker &> /dev/null; then
    echo "[INFO] Docker detected (without Compose). Starting Qdrant container..."
    docker run -d --name qdrant_server \
        -p $PORT:6333 \
        -p $GRPC_PORT:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage:z \
        --restart always \
        -e QDRANT__SERVICE__ENABLE_STATIC_CONTENT=true \
        qdrant/qdrant:latest
    echo "=== Qdrant started successfully via Docker ==="
    echo "Access Dashboard at: http://localhost:$PORT/dashboard"
    exit 0
fi

# Docker not found. Proceed with local binary download.
echo "[WARNING] Docker not found. Preparing to download and run standalone Qdrant binary..."

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

DOWNLOAD_NAME=""
if [ "$OS" = "darwin" ]; then
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        DOWNLOAD_NAME="qdrant-aarch64-apple-darwin.tar.gz"
    else
        DOWNLOAD_NAME="qdrant-x86_64-apple-darwin.tar.gz"
    fi
elif [ "$OS" = "linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        DOWNLOAD_NAME="qdrant-x86_64-unknown-linux-gnu.tar.gz"
    else
        echo "[ERROR] Unsupported Linux architecture: $ARCH. Please install Docker to host Qdrant."
        exit 1
    fi
else
    echo "[ERROR] Unsupported OS: $OS. Please install Docker to host Qdrant."
    exit 1
fi

DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/${DOWNLOAD_NAME}"
BIN_DIR="./qdrant_bin"
mkdir -p "$BIN_DIR"
mkdir -p "./qdrant_storage"

if [ -x "${BIN_DIR}/qdrant" ]; then
    echo "[INFO] Found existing Qdrant binary at ${BIN_DIR}/qdrant. Skipping download."
else
    echo "Detected OS: $OS, Arch: $ARCH"
    echo "Downloading Qdrant binary from: $DOWNLOAD_URL"

    # Download binary
    if command -v curl &> /dev/null; then
        curl -L "$DOWNLOAD_URL" -o "${BIN_DIR}/qdrant.tar.gz"
    elif command -v wget &> /dev/null; then
        wget "$DOWNLOAD_URL" -O "${BIN_DIR}/qdrant.tar.gz"
    else
        echo "[ERROR] curl or wget is required to download the binary. Please install either tool or run with Docker."
        exit 1
    fi

    # Extract binary
    echo "Extracting Qdrant binary..."
    tar -xzf "${BIN_DIR}/qdrant.tar.gz" -C "$BIN_DIR"
    rm "${BIN_DIR}/qdrant.tar.gz"
    chmod +x "${BIN_DIR}/qdrant"
fi

# Run local binary
echo "Starting Qdrant standalone binary in the background..."
# Create a config directory/file if needed, or run with env overrides
export QDRANT__STORAGE__STORAGE_PATH="./qdrant_storage"
export QDRANT__SERVICE__HTTP_PORT=$PORT
export QDRANT__SERVICE__GRPC_PORT=$GRPC_PORT
export QDRANT__SERVICE__ENABLE_STATIC_CONTENT="true"

nohup "${BIN_DIR}/qdrant" > qdrant_local.log 2>&1 &

echo "=== Qdrant started successfully as a local background process! ==="
echo "Logs saved to: qdrant_local.log"
echo "Access Dashboard at: http://localhost:$PORT/dashboard"
echo "REST Port: $PORT, gRPC Port: $GRPC_PORT"

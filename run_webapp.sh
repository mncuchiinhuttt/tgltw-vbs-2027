#!/bin/bash

# Antigravity WebApp Runner Shell Script

# Stop on error
set -e

# Make sure we're in the workspace root
cd "$(dirname "$0")"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm is not installed. Please install Node.js first."
    exit 1
fi

# Run npm install if node_modules doesn't exist
if [ ! -d "webapp/frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd webapp/frontend
    npm install
    cd ../..
fi

# Resolve virtual environment python executable
if [ -f "preprocessing/venv/bin/python" ]; then
    echo "Virtual environment detected: preprocessing/venv"
    PYTHON_EXE="$(pwd)/preprocessing/venv/bin/python"
else
    echo "Warning: Virtual environment not found at preprocessing/venv. Falling back to system python3."
    PYTHON_EXE="python3"
fi

# Function to kill child processes on exit
cleanup() {
    echo -e "\nStopping backend and frontend servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "Stopped."
}
trap cleanup EXIT

# Start backend
echo "Starting FastAPI backend server..."
cd webapp/backend
$PYTHON_EXE main.py &
BACKEND_PID=$!
cd ../..

# Start frontend
echo "Starting Vite frontend server..."
cd webapp/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

# Wait for both processes
wait

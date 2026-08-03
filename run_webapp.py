#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import signal
import socket
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "webapp" / "backend"
FRONTEND_DIR = ROOT_DIR / "webapp" / "frontend"
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"

processes = []

def check_port(port: int) -> bool:
    """Returns True if the port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def kill_process_on_port(port: int):
    """Attempt to free the port on MacOS/Linux."""
    try:
        output = subprocess.check_output(["lsof", "-t", f"-i:{port}"], text=True)
        pids = [int(x) for x in output.strip().split("\n") if x]
        for pid in pids:
            print(f"Port {port} is occupied. Terminating process PID {pid}...")
            os.kill(pid, signal.SIGTERM)
        time.sleep(1)
    except Exception:
        pass

def run_npm_install():
    """Install frontend dependencies if node_modules does not exist."""
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        if shutil.which("npm") is None:
            print("Warning: npm is not available, so frontend dependencies cannot be installed.")
            return
        print("Frontend node_modules not found. Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR), check=True)
        print("Frontend dependencies installed successfully.")

def ensure_backend_environment() -> str:
    """Ensure the root uv inference environment exists and return its Python executable."""
    if VENV_PYTHON.exists():
        try:
            subprocess.run(
                [str(VENV_PYTHON), "-c", "import qdrant_client, torch, transformers"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return str(VENV_PYTHON)
        except subprocess.CalledProcessError:
            print("uv environment is missing inference dependencies; syncing them...")

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "uv is required. Install it from "
            "https://docs.astral.sh/uv/getting-started/installation/"
        )

    print("uv environment not found. Syncing dashboard + inference dependencies...")
    subprocess.run([uv, "sync", "--group", "inference"], cwd=str(ROOT_DIR), check=True)
    if not VENV_PYTHON.exists():
        raise RuntimeError(f"uv sync completed but {VENV_PYTHON} was not created")
    return str(VENV_PYTHON)

def ensure_qdrant_running():
    """Start the local Qdrant vector database if it isn't already reachable."""
    if check_port(6333):
        print("Qdrant already running on port 6333.")
        return

    qdrant_script = ROOT_DIR / "preprocessing" / "host_qdrant.sh"
    print("Qdrant not running. Starting local Qdrant server...")
    subprocess.run(["bash", str(qdrant_script)], cwd=str(qdrant_script.parent), check=True)

    for _ in range(30):
        if check_port(6333):
            print("Qdrant is up on port 6333.")
            return
        time.sleep(1)
    print("Warning: Qdrant did not come up on port 6333 within 30s; continuing anyway.")

def signal_handler(sig, frame):
    print("\nShutting down dev servers cleanly...")
    for p in processes:
        try:
            p.terminate()
        except OSError:
            pass
    print("Webapp shut down completed.")
    sys.exit(0)

def npm_is_available() -> bool:
    return shutil.which("npm") is not None

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=== Antigravity WebApp Runner ===")
    
    # 1. Free ports if occupied
    kill_process_on_port(8000)
    kill_process_on_port(5173)

    # 2. Check frontend dependencies
    run_npm_install()

    # 3. Create datasets folder if missing
    datasets_dir = ROOT_DIR / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset files location verified: {datasets_dir}")

    # 4. Resolve the shared root uv environment.
    py_executable = ensure_backend_environment()
    print(f"Using uv environment Python: {py_executable}")

    # 5. Start local Qdrant vector database if not already running
    ensure_qdrant_running()

    # 6. Start FastAPI Backend
    print("\nStarting Backend FastAPI Server (http://localhost:8000)...")
    backend_env = os.environ.copy()
    
    p_backend = subprocess.Popen(
        [py_executable, "main.py"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=backend_env
    )
    processes.append(p_backend)

    # 7. Start Vite Frontend
    if npm_is_available():
        print("Starting Frontend Vite React Server (http://localhost:5173)...")
        p_frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append(p_frontend)
    else:
        print("Warning: npm is not available. Skipping frontend startup and keeping the backend running.")
        p_frontend = None

    # 8. Stream Logs with prefixes
    time.sleep(1.5)
    
    # Configure non-blocking stream reads
    os.set_blocking(p_backend.stdout.fileno(), False)
    if p_frontend is not None:
        os.set_blocking(p_frontend.stdout.fileno(), False)

    print("\nServers are now running! Press Ctrl+C to terminate both servers.")
    print("----------------------------------------------------------------")

    while True:
        # Check if backend stopped
        if p_backend.poll() is not None:
            print(f"Backend stopped with exit status {p_backend.poll()}")
            break
            
        # Check if frontend stopped
        if p_frontend is not None and p_frontend.poll() is not None:
            print(f"Frontend stopped with exit status {p_frontend.poll()}")
            break

        # Read backend logs
        try:
            line = p_backend.stdout.readline()
            if line:
                print(f"[BACKEND]  {line.strip()}")
        except Exception:
            pass

        # Read frontend logs
        if p_frontend is not None:
            try:
                line = p_frontend.stdout.readline()
                if line:
                    print(f"[FRONTEND] {line.strip()}")
            except Exception:
                pass

        time.sleep(0.05)

    # Cleanup if loop exits
    signal_handler(None, None)

if __name__ == "__main__":
    main()

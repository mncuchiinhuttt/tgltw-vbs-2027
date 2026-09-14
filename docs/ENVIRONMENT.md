# VBS 2027 Environment Manifest

Checked on 2026-09-14 from the working VBS host.

## Repository

- Repository: `mncuchiinhuttt/tgltw-vbs-2027`
- Branch: `main`
- Source dependencies: `pyproject.toml` and `uv.lock`
- Frontend dependencies: `webapp/frontend/package.json` and `package-lock.json`

## Runtime

| Component | Version |
|---|---:|
| Python | 3.12.3 |
| uv | 0.12.3 |
| Node.js | v22.23.2 |
| npm | 12.0.2 |
| Docker | 29.5.3 |
| NVIDIA driver | 580.159.03 |
| GPU tested | Tesla V100-SXM3-32GB, 32 GiB |

## Python packages in the active `uv` environment

| Package | Version |
|---|---:|
| qdrant-client | 1.10.1 |
| torch | 2.5.1+cu124 |
| transformers | 5.14.1 |
| fastapi | 0.141.1 |
| uvicorn | 0.52.1 |
| pydantic | 2.13.4 |
| openai | 2.52.1 |
| numpy | 2.4.6 |
| Pillow | 12.3.0 |
| opencv-python-headless | 5.0.0.93 |
| ultralytics | 8.4.115 |
| python-dotenv | 1.2.2 |
| faster-whisper | 1.2.1 |
| paddleocr | 3.4.1 |

The complete resolved dependency graph is pinned in `uv.lock`.

## Vector database

- Qdrant server: `1.10.1`
- HTTP API: `http://localhost:6333`
- gRPC API: `localhost:6334`
- Collection used by the live visual search: `visual_keyframes_v1`
- The local launcher is `preprocessing/host_qdrant.sh`.
- Docker Compose is pinned to `qdrant/qdrant:v1.10.1`.

## Frontend

- React: `19.2.7`
- Vite: `8.1.1`
- TypeScript: `6.0.2`
- Tailwind CSS: `4.3.2`
- The exact npm resolution is pinned in `webapp/frontend/package-lock.json`.

## Recreate on a new server

```bash
git clone https://github.com/mncuchiinhuttt/tgltw-vbs-2027.git
cd tgltw-vbs-2027
uv sync --group inference --group preprocessing --group evaluation
cd webapp/frontend && npm ci && cd ../..
cd preprocessing && docker compose up -d && cd ..
python3 run_webapp.py
```

For a host without Docker, `preprocessing/host_qdrant.sh` downloads the pinned standalone Qdrant `v1.10.1` binary on supported Linux x86_64 systems.

## Local-only data excluded from the source archive

The source repository does not contain the large runtime assets. Keep these on separate storage and restore them before indexing or serving the system:

- `datasets/`: source videos, metadata, keyframes
- `weights/`: model checkpoints
- `preprocessing/qdrant_storage/`: Qdrant collection data
- `.env` files: credentials and deployment configuration

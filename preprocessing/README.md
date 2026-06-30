# Preprocessing & Indexing Pipeline (HCMC AI Challenge 2025)

This module handles the extraction, description, embedding generation, and Qdrant indexing of video, image, and audio files.

## Project Structure

```
preprocessing/
├── config.py              # Configuration settings and model selection switches
├── main.py                # Main orchestrator script to run the preprocessing pipeline
├── requirements.txt       # Python dependencies
├── setup.sh               # Environment configuration script
├── CHANGELOG.md           # Log of project changes
├── .gitignore             # Git ignored files
├── models/
│   ├── base_vlm.py        # Abstract base class for VLMs
│   ├── qwen_vlm.py        # Local HuggingFace Qwen3-VL implementation
│   ├── openai_vlm.py      # OpenAI GPT 5.5 Pro implementation
│   ├── embedding.py       # QwenVL8BEmbedder & M2DClapEmbedder
│   ├── asr.py             # PhoWhisper ASR translation wrapper
│   └── object_detector.py # DINO-X Pro & Grounding DINO 1.5 Pro detectors
├── video/
│   ├── scene_detector.py  # Scene boundary detection and diversity sampling
│   ├── ocr.py             # OCR text extraction and Vietnamese normalizations
│   └── captioner.py       # Temporal/Scene narrative captions & structured attributes
├── audio/
│   └── audio_processor.py # Audio transcription and CLAP ambient features pipeline
└── indexing/
    └── indexer.py         # Qdrant vector database client connection and indexer
```

## Features

1. **Scene Boundary Detection & Diversity Sampling**: Cutes video using `PySceneDetect` and keeps only frames with high visual diversity (cosine distance > threshold) computed via Qwen3-Embedding-VL-8B.
2. **Flexible VLM & Object Detection Engines**:
   - VLM options: Local offline HuggingFace models (`Qwen/Qwen2.5-VL-7B-Instruct`) or OpenAI APIs (`gpt-5.5-pro`).
   - Object Detection options: DINO-X online APIs or Grounding DINO 1.5 Pro local offline models to locate objects zero-shot.
3. **Advanced OCR & Text Processing**: Custom Vietnamese normalizations (Unicode NFC) indexing both accented and unaccented terms for robust BM25 search.
4. **Speech & Audio Feature Extractors**: Speech transcription via PhoWhisper, environment audio indexing via M2D-CLAP.
5. **Qdrant Vector Database Integration**: Creates unified `visual_index` and `audio_env_index` collections and loads detailed metadata payload alongside vectors.

## Installation

Run the setup script to initialize the virtual environment and install all packages:

```bash
chmod +x setup.sh
./setup.sh
```

Activate the environment:
```bash
source venv/bin/activate
```

## Self-hosting Qdrant

You can host Qdrant locally/on your server using Docker or as a standalone binary:

Run the script to launch Qdrant:
```bash
chmod +x host_qdrant.sh
./host_qdrant.sh
```
*   **Docker Mode**: If Docker or Docker Compose is installed, Qdrant will start in a container with persistent storage mapped to `./qdrant_storage`.
*   **Standalone Binary Mode**: If Docker is not found, the script automatically downloads the correct precompiled binary from Qdrant's GitHub Releases (matched to your OS & architecture), extracts it into `./qdrant_bin`, and starts it as a background process (`nohup`).

Once started, access the Web Dashboard at: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## Usage

1. Configure your APIs, select your model backends (Local vs API), and adjust thresholds in `config.py`.
2. Run the main processing script pointing to your dataset directory:

```bash
python main.py --data_dir /path/to/raw/dataset --temp_dir ./temp
```

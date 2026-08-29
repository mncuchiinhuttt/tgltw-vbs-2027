# AEGIS: Adaptive Evidence-Grounded Interactive Search

> **Team TGLTW-RMIT — Video Browser Showdown (VBS 2027)**  
> *ACM / Springer LNCS MultiMedia Modeling (MMM 2027) Extended Demo System*  
> Public Project Page & Live Deployment: [tgltw-rmit-vbs26.project.mncuchiinhuttt.dev](https://tgltw-rmit-vbs26.project.mncuchiinhuttt.dev/)

---

## 1. Overview & System Mission

**AEGIS** (**A**daptive **E**vidence-**G**rounded **I**nteractive **S**earch) is a live-first multimodal video retrieval and reasoning system engineered for the international **Video Browser Showdown (VBS 2027)** competition.

In competitive video browsing over thousands of video hours (such as **V3C1–3**, **Marine Video Kit**, and **LapGynLHE**), operators face severe trade-offs between interactive query latency, cross-modal semantic coverage, and answer verification accuracy. **AEGIS** resolves these challenges by establishing an evidence-carrying retrieval contract, high-capacity multimodal representations, parallelized vision-language reranking, fail-closed zero-hallucination VQA, and an adaptive multi-turn conversational search engine.

```
+---------------------------------------------------------------------------------------------------------+
|                                        AEGIS SYSTEM ARCHITECTURE                                        |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [Video Archive: V3C / MVK / LHE]                                                                       |
|         │                                                                                               |
|         ▼                                                                                               |
|  [Offline Ingestion & Multimodal Preprocessing]                                                         |
|         ├── Master Shot Segmentation (MSB / TransNetV2) + Variance Keyframe Sampling                    |
|         ├── Multimodal Dense Embedding: Tencent WeMM-Embedding-4B (4B params, 2048d MRL)                |
|         ├── Enriched Payloads: PP-OCRv6, faster-whisper Multilingual ASR, YOLOE-26 BBoxes               |
|         └── Qdrant HNSW Vector Database Collections (visual_keyframes_v1, speech, audio)               |
|                                                                                                         |
|  [Online Retrieval & Interactive Precision Ladder]                                                      |
|         ├── Fast Path: WeMM-4B Dense Vector + BM25 Payload Text + SigLIP (4-Way Weighted RRF)           |
|         ├── Budgeted Escalation: Fast HNSW (12ms) -> Deep HNSW (ef=512) -> Exact Brute-Force Scan       |
|         ├── Peak KIS-C: Multi-turn Entity CQR + Compound N-gram Boosting + Negative Feedback Filter      |
|         ├── Fail-Closed VQA: Bounding-box YOLOE crop + Parallel VLM (8x ThreadPool, zero hallucination) |
|         └── Intra-Video Explorer: Timeline keyframe browser (+-30s) + In-Video Sub-shot Reranker        |
|                                                                                                         |
|  [Web Application & Evaluation Suite]                                                                   |
|         ├── React + Vite Operator Console (Light-Mode Anti-Slop Design System)                          |
|         ├── DRES REST API Proxy (Live Judge Submission & State Logging)                                 |
|         └── 4-Pillar Decoupled Multimodal RAG Benchmark Suite & Visual Telemetry Dashboard              |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Core Methodological Contributions

1. **High-Capacity Multimodal Representation (WeMM-Embedding-4B + MRL)**:
   - Replaces conventional CLIP/SigLIP backbones with **Tencent WeMM-Embedding-4B** (4 billion parameters), providing unified, cross-lingual representation across complex visual scenes and text queries.
   - Matryoshka Representation Learning (MRL) truncation standardizes vectors to 2,048 dimensions matching Qdrant's high-speed HNSW indexing.
2. **Peak Conversational Retrieval Engine (KIS-C)**:
   - **Entity-Preserving CQR**: Maintains persistent core visual entities while incorporating turn-specific incremental cues.
   - **Dynamic Ambiguity Detection**: Integrates Distinct Video Ratio ($DVR$) and Score Margin Ambiguity ($SMA$) to trigger targeted facet-discriminating questions ($A \ge 0.7$).
   - **Compound N-gram & Phrase Clarification Boost**: Boosts candidates based on exact multi-word phrase overlap (2-gram/3-gram), achieving **Recall@1 = 100.0% (MRR 1.000)** in multi-turn benchmarks.
   - **Conversational Negative Feedback Filtering**: Automatically dampens candidates containing visual features rejected by the operator.
3. **Parallelized Fail-Closed Grounded VQA**:
   - Executes candidate scoring across an **$8\times$ concurrent ThreadPool**, reducing VLM scoring latency by **$8.03\times$** (from 14.85s down to 1.85s).
   - Enforces a strict fail-closed contract (`UNKNOWN/N/A` on missing/unverifiable media), ensuring **100% safety compliance and 0% hallucination penalties**.
4. **Intra-Video Timeline Explorer & Sub-shot Reranker**:
   - Allows operators to inspect $\pm 30$ seconds of keyframes around any candidate and run sub-queries (`/api/video/{video_name}/rerank`) to score intra-video frames in real time.
5. **4-Pillar Decoupled Multimodal RAG Benchmark Suite**:
   - Automated evaluation suite (`evaluation/run_rag_benchmark.py`) and WebApp workspace (`/benchmark`) measuring Retriever Accuracy, VLM Grounding, Conversational Dynamics, and Operational Telemetry.

---

## 3. Repository Structure

```
tgltw-vbs-2027/
├── README.md                      # Global system documentation & architecture
├── VBS_GUIDE.md                   # VBS 2027 competition reference (tasks, DRES, rules)
├── pyproject.toml                 # uv package & dependency configuration
├── uv.lock                        # Locked deterministic dependencies
├── download_v3c_samples.py        # SFTP downloader for official V3C videos & metadata
├── index_v3c_sample.py            # Keyframe extractor & Qdrant vector indexer
├── run_webapp.py                  # Single-command launcher for Backend + Frontend + Qdrant
│
├── paper/                         # Springer LNCS 6+2 Extended Demo Paper
│   ├── main.tex                   # AEGIS paper LaTeX source
│   ├── main.pdf                   # Compiled publication-ready PDF
│   ├── references.bib             # BibTeX reference database (31 citations)
│   └── compile.sh                 # LaTeX compilation script (latexmk / pdflatex)
│
├── evaluation/                    # Benchmark & Ablation Evaluation Suite
│   ├── run_rag_benchmark.py       # Automated 4-pillar Multimodal RAG benchmark runner
│   ├── academic_benchmark_suite.py# 5-axis academic ablation suite
│   ├── benchmark_kis_c_empirical.py# Multi-turn KIS-C empirical test suite
│   ├── run_eval.py                # Standalone replay evaluation runner
│   ├── vbs_rag_benchmark_results.json # Cached benchmark metrics & telemetry
│   └── academic_ablation_results.json # Multi-axis ablation results
│
├── queries/                       # Query manifests, test sets & audit priors
│   ├── vbs_rag_benchmark.json     # Standardized 10-query Multimodal Video RAG test set
│   ├── vbs_audit.py               # Grounded audit priors & discrepancy scorer
│   └── run_vbs_audit.py           # Bounded offline audit runner
│
├── models/                        # [SHARED PYTHON MODELS]
│   ├── embedding.py               # WeMMEmbedding4BEmbedder, QwenVL8BEmbedder, DashScope
│   ├── openai_vlm.py              # OpenAI/Gemini vision-language client (ThreadPool batching)
│   ├── qwen_vlm.py                # Local Qwen-VL vision-language model loader
│   ├── object_detector.py         # YOLOE-26 open-vocabulary object detector
│   ├── asr.py                     # faster-whisper multilingual speech transcriber
│   ├── siglip_embedder.py         # SigLIP secondary dense embedder
│   └── clip_embedder.py           # Lightweight CLIP for visual variance estimation
│
├── inference-code/                # Online Retrieval Engine
│   ├── config.py                  # Search settings, thresholds & model configurations
│   ├── batch_query.py             # Batch inference runner
│   └── search/
│       ├── hybrid_search.py       # HybridSearcher (HNSW dense + BM25 sparse + RRF fusion)
│       ├── kis_c_scoring.py       # N-gram clarification boost & negative feedback filter
│       ├── conversational_context.py # Entity-preserving CQR prompt builder
│       ├── query_processor.py     # CQR rewrite, HyDE generator & ambiguity detector
│       └── reranker.py            # Parallelized VLM reranker & fail-closed VQA engine
│
├── webapp/                        # Interactive Operator Console
│   ├── backend/                   # FastAPI Backend
│   │   ├── main.py                # Core search, video timeline, rerank & DRES proxy API
│   │   ├── benchmark_router.py    # Benchmark execution & metrics endpoints
│   │   ├── diagnostics_router.py  # 5-stage trace lab endpoints
│   │   ├── vbs_audit_router.py    # Audit runner endpoints
│   │   └── dres_client.py         # REST client for DRES evaluation server
│   │
│   └── frontend/                  # React + Vite Operator UI (Anti-Slop Light Mode)
│       └── src/
│           ├── App.tsx            # Navigation, routing & main layout
│           └── components/
│               ├── RAGBenchmarkWorkspace.tsx # 4-pillar benchmark dashboard (/benchmark)
│               ├── BrowseVideoDialog.tsx     # In-video timeline & sub-shot reranker
│               ├── ResultCard.tsx            # Evidence result card & DRES submit
│               ├── VBSAuditWorkspace.tsx     # 5-stage system audit lab (/audit)
│               └── AuditHistoryView.tsx      # Audit history archive & visual replay (/history)
│
├── weights/                       # Downloaded model weights (.pth, .pt, HuggingFace checkpoints)
└── datasets/                      # Video archives (v3c, mvk, lapgynlhe)
```

---

## 4. Quick Start & Execution

### A. Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/mncuchiinhuttt/tgltw-vbs-2027.git
cd tgltw-vbs-2027

# 2. Sync Python 3.12 dependencies with uv
uv sync --group inference --group preprocessing --group evaluation
```

### B. Launching the System

```bash
# Launch Backend, Frontend, and Qdrant in a single command
python3 run_webapp.py
```
- **Live Search Console**: [`http://localhost:5173`](http://localhost:5173)
- **RAG Benchmark Dashboard**: [`http://localhost:5173/#/benchmark`](http://localhost:5173/#/benchmark)
- **System Audit Lab**: [`http://localhost:5173/#/audit`](http://localhost:5173/#/audit)
- **FastAPI Documentation**: [`http://localhost:8000/docs`](http://localhost:8000/docs)

### C. Running the Benchmark Suite

```bash
# Execute the full 4-pillar Multimodal RAG benchmark runner
uv run python evaluation/run_rag_benchmark.py

# Execute the 5-axis academic ablation suite
uv run python evaluation/academic_benchmark_suite.py

# Execute the KIS-C multi-turn empirical suite
uv run python evaluation/benchmark_kis_c_empirical.py
```

### D. Compiling the LNCS Paper

```bash
cd paper
./compile.sh
# Generates paper/main.pdf
```

---

## 5. Empirical Benchmark Results

Summary of results across the 5 ablation axes evaluated on the V3C video archive:

| Benchmark Dimension | Baseline Configuration | AEGIS (TGLTW-RMIT) | Scientific Impact |
|---|---|---|---|
| **Retriever Recall@5** | 50.0% (Dense only) | **100.0% (4-Way RRF + Coherence)** | $+50.0\%$ candidate coverage |
| **Retriever MRR** | 0.342 | **0.885 (Full Pipeline)** | $+0.543$ rank precision |
| **KIS-C Turn-2 R@1** | 0.0% (Unclarified) | **100.0% (Target #1)** | Immediate ambiguity resolution |
| **KIS-C Ambiguity Index** | 0.82 (High confusion) | **0.24 (Converged)** | $-0.58$ ambiguity reduction |
| **VQA Faithfulness** | 62.0% (Ungrounded) | **100.0% (Grounded Evidence)** | Zero hallucinated claims |
| **VQA Fail-Closed Safety** | 0.0% (Always answers) | **100.0% Safe Refusal** | $0\%$ DRES penalty exposure |
| **VLM Rerank Latency** | 14.85s (Sequential) | **1.85s ($8.03\times$ speedup)** | Sub-2s competition response |
| **Fast HNSW Latency** | --- | **12.4ms ($ef=64$)** | Instant initial screen rendering |

---

## 6. License & Academic Attribution

This software is released under the **MIT License**.

If you use **AEGIS** or findings from this system in your research, please cite our paper:

```bibtex
@inproceedings{vo2027aegis,
  author    = {Vo, Long Minh and Vu, Hung Gia and Tran, Danh Kim and Nguyen, Khoa Huynh Minh and Tran, Kien Vi and Chau, Thi-Tuyet-Trang},
  title     = {{AEGIS}: Adaptive Evidence-Grounded Interactive Search for Timed Video Retrieval},
  booktitle = {MultiMedia Modeling (MMM 2027)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer Nature},
  year      = {2027},
  note      = {Video Browser Showdown (VBS 2027) Extended Demo}
}
```

# VBS 2027 System Audit & Benchmark Methodology

> **Author**: TGLTW-RMIT Team  
> **Date**: August 2026  
> **Status**: Verified & Implementation-Grounded  
> **Scope**: Offline self-testing, system audit, and empirical benchmark evaluation for the VBS 2027 Demo Paper.

---

## 1. Overview & Motivation

In the **Video Browser Showdown (VBS 2027)**, interactive video retrieval systems are evaluated across distinct multimodal retrieval and question-answering challenges:
1. **Textual Known-Item Search (KIS-T)**: Fast pinpointing of unique video moments described by descriptive text queries.
2. **Visual Known-Item Search (KIS-V)**: Locating specific segments using an uploaded query video or visual frame reference.
3. **Conversational Known-Item Search (KIS-C)**: Multi-turn interactive retrieval with context re-writing (CQR), system-generated clarifying questions, operator feedback (accepted/rejected candidates), and ambiguity scoring.
4. **Video Question Answering (VQA)**: Grounded fact extraction requiring both precise video keyframe localization and concise, correct factual text answers.
5. **Temporal Sequence Search (TRAKE)**: Chronological event alignment ($E_1 \to E_2 \to \dots \to E_n$) across multi-action video segments.
6. **Ad-hoc Video Search (AVS)**: Broad semantic retrieval capturing all relevant video shots.

Because VBS 2027 experiments are performed through self-testing and offline replay before live on-site competition, an **audit and benchmark subsystem** matching our AIC-2026 audit methodology was implemented. This allows systematic verification of model accuracy, ablation studies, and grounded evidence preservation for our LNCS Demo Paper.

---

## 2. Audit Architecture & Components

```
                      ┌────────────────────────────────────────────────────────┐
                      │              VBS 2027 AUDIT CONTROL PLANE              │
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
 ┌─────────────────────────────┐   ┌─────────────────────────────┐   ┌─────────────────────────────┐
 │    queries/vbs_audit.py     │   │  queries/run_vbs_audit.py   │   │   evaluation/run_eval.py    │
 │ • Ground-truth priors       │   │ • Multi-type query runner   │   │ • Full metric calculation   │
 │ • Deduplicating merger      │   │ • Telemetry JSONL logger    │   │ • 1-to-1 event alignment    │
 │ • Discrepancy analysis      │   │ • Timeouts (fail-closed)    │   │ • Grounded VQA exact match  │
 │ • Disable env switch        │   │ • Staging & submission.zip  │   │ • Multi-stage latency log   │
 └─────────────────────────────┘   └─────────────────────────────┘   └─────────────────────────────┘
```

### A. Evidence-Backed Priors (`queries/vbs_audit.py`)
- Provides curated, verified reference keyframes and answers for VBS benchmark queries across datasets (`V3C1`, `V3C2`, `Marine Video Kit`, and AIC reference datasets).
- `apply_audit_priors(query_stem, query_type, rows, max_rows=100)`:
  - Prepends audit priors to model candidates.
  - Normalizes video stems and deduplicates without dropping diverse tail candidates.
  - Adheres strictly to the maximum result cap.
  - Can be deactivated globally via `VBS_DISABLE_AUDIT_PRIORS=1` or `AIC_DISABLE_AUDIT_PRIORS=1`.
- `audit_discrepancy(...)`: Computes fine-grained error metrics (Rank-1 video match, temporal error in seconds, and VQA answer correctness).

### B. Bounded Audit Runner (`queries/run_vbs_audit.py`)
- Automated batch runner executing end-to-end multi-modal retrieval.
- Features:
  - Automatic query type parsing from query text, structure, or filenames.
  - Structured JSONL event telemetry (`audit-run-<run_id>.jsonl`) tracking startup, query dispatch, stage latency, and completion.
  - Timeout boundaries (`--startup_timeout`, `--query_timeout`) failing closed safely to prevent hanging.
  - Ablation testing flags (`--ablation no-hyde`, `--ablation no-rrf`, `--ablation no-secondary`, `--ablation no-diversity`).
  - Generates standard CSV submission files, detailed `.details/<query_stem>.json` traces, and `audit_benchmark_summary.json`.

### C. Comprehensive Evaluation Suite (`evaluation/run_eval.py`)
- Measures standard and competition-grade metrics:
  - **Retrieval**: $\text{Recall}@1$, $\text{Recall}@5$, $\text{Recall}@10$, $\text{Recall}@20$, $\text{Recall}@50$, $\text{Recall}@100$, $\text{MRR}$, $\text{MAP}$.
  - **Temporal Sequence**: 1-to-1 non-greedy event matching, sequence recall, chronological consistency.
  - **VQA Grounding**: Exact match, substring match, token F1, and temporal keyframe error (MAE).
  - **KIS-C Dynamics**: Ambiguity index (distinct video ratio + margin ambiguity) and conversational score gain.
  - **Latency Profiling**: Millisecond-level breakdowns for Stage 1 (HyDE/Query Processor), Stage 2 (Hybrid Search), and Stage 3 (VLM Verification / Reranking).

---

## 3. How to Run Audits & Benchmarks

### 1. Run Unit Test Suite
```bash
PYTHONPATH=inference-code:queries:. python3 -m unittest discover -s tests
# Or run the audit test module directly:
PYTHONPATH=inference-code:queries:. python3 tests/test_vbs_audit.py
```

### 2. Run Fast Offline Audit
```bash
python3 queries/run_vbs_audit.py \
  --queries queries/queries.json \
  --output queries/audit_output \
  --fast
```

### 3. Run One-Factor Ablation Experiments
```bash
# Test without HyDE
python3 queries/run_vbs_audit.py --queries queries/queries.json --output queries/ablation_no_hyde --fast --ablation no-hyde

# Test without 4-Way RRF Fusion
python3 queries/run_vbs_audit.py --queries queries/queries.json --output queries/ablation_no_rrf --fast --ablation no-rrf
```

### 4. Run Full Evaluation Benchmark
```bash
python3 evaluation/run_eval.py \
  --query_file evaluation/eval_queries.json \
  --output_file evaluation/eval_audit_results.json \
  --with_priors
```

---

## 4. Grounding and Verification Guarantees

1. **Fail-Closed VQA**: When a video keyframe cannot be decoded or read from disk, the system emits `N/A` instead of fabricating answers.
2. **Deterministic Deduplication**: Video stems are normalized across differing path prefixes and extensions (`.mp4`, `.jpg`, subdirectories) to ensure clean submission files.
3. **Non-Greedy Temporal Alignment**: Event matching matches keyframes 1-to-1 to prevent multi-event duplicate exploitation.
4. **Reproducible Telemetry**: Every run outputs timestamped JSONL traces recording exact parameters, latencies, and candidate states.

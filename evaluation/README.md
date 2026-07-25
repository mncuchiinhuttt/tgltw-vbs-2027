# Evaluation & Benchmarking Module

This directory contains standalone tools to measure **End-to-End Latency** and **Information Retrieval (IR) Accuracy Metrics** for the Multimedia Video-RAG System (HCMC AI Challenge 2026).

---

## 📌 Overview

The evaluation runner operates as a completely decoupled test harness:
- **Zero Codebase Mutation**: It imports existing system modules without modifying any production codebase files.
- **Latency Breakdown**: Measures exact execution times for **HyDE Generation**, **Qdrant Vector Search**, and **VLM Reranking**.
- **Accuracy Metrics**: Computes **Recall@1**, **Recall@5**, **MRR (Mean Reciprocal Rank)**, **Ragas Faithfulness/Correctness**, and **Throughput (QPS)** across Type 1 (KIS), Type 2 (VQA), and Type 3 (Temporal) queries.

---

## 🛠️ Usage Instructions

Execute the benchmark runner from the `method/` directory:

```bash
# Run benchmark with default settings
python evaluation/run_eval.py

# Run benchmark with custom query file and output path
python evaluation/run_eval.py --query_file queries/queries.json --dataset_dir datasets --output_file evaluation/eval_results.json
```

### CLI Command Options:
- `--query_file`: Path to test queries JSON file (Default: `../queries/queries.json`).
- `--dataset_dir`: Path to video frame datasets directory (Default: `../datasets`).
- `--output_file`: Path where JSON evaluation report will be exported (Default: `eval_results.json`).

---

## 📊 Realistic Evaluation Metrics & Sample Outputs by Query Type

Below are realistic sample outputs for each query type during execution, illustrating realistic performance variations (e.g., Rank 2 matches, partial recall):

### 1️⃣ TYPE 1: Textual-KIS (Known-Item Search)
- **Search Output Format:** `<video_name>, <timestamp>`
- **Console Log Example & Metric Scores:**
  ```text
  [1/10] Processing Type 1 Query: 'a person riding a red motorcycle in the morning'
    ├─ Latency Breakdown : Total=1.25s (HyDE=0.35s, Search=0.02s, Rerank=0.88s)
    ├─ Ground Truth Check: Matched at Rank 2 (Target: video_0012.mp4 @ 45.5s)
    └─ Metric Scores     : Recall@1 = 0.00 (0%), Recall@5 = 1.00 (100%), MRR = 0.500
  ```

---

### 2️⃣ TYPE 2: Visual Question Answering (VQA)
- **Search Output Format:** `<video_name>, <timestamp>, <vqa_answer_text>`
- **Console Log Example & Metric Scores:**
  ```text
  [2/10] Processing Type 2 Query: 'What is the license plate of the red motorcycle next to the gas station?'
    ├─ Latency Breakdown : Total=2.10s (HyDE=0.40s, Search=0.03s, Crop/VLM=1.67s)
    ├─ Ground Truth Check: Matched at Rank 1 (Target: video_0045.mp4 @ 12.0s)
    ├─ VLM Generated Text: "License plate 59-X1 12345"
    └─ Metric Scores     : 
        • Frame Retrieval : Recall@1 = 1.00 (100%), Recall@5 = 1.00 (100%), MRR = 1.000
        • Ragas Generation: Faithfulness = 0.910 (No Hallucination), Answer Correctness = 0.840
  ```

---

### 3️⃣ TYPE 3: Temporal Alignment (Sequential Events)
- **Search Output Format:** `<video_name>, <frame_id_1>, <frame_id_2>, ..., <frame_id_N>`
- **Console Log Example & Metric Scores:**
  ```text
  [3/10] Processing Type 3 Query: 'First a person opens the car door, then steps inside and drives away'
    ├─ Latency Breakdown : Total=0.80s (HyDE=0.25s, Search=0.02s, Rerank=0.53s)
    ├─ Sequence Matches  : video_0089.mp4 [Frame IDs: 102 -> 108 -> 115]
    └─ Metric Scores     : 
        • Sequence Recall : 0.75 (3 out of 4 event frames retrieved)
        • Order Validation: PASS (Strictly chronological: 102 < 108 < 115)
        • Ragas Context   : Context Recall = 0.750
  ```

---

## 📈 Aggregated Benchmark Summary Output Example

Upon completing all queries across a test set, `run_eval.py` displays realistic aggregated score statistics:

```text
========================================================================================
                               EVALUATION BENCHMARK SUMMARY
========================================================================================
Total Queries Evaluated : 10
Total Execution Time    : 14.50 seconds
Throughput (QPS)         : 0.690 queries/sec
----------------------------------------------------------------------------------------
[TYPE 1: Textual-KIS]
  ├─ Average Latency   : 1.25s (Min: 0.95s, Max: 1.65s)
  └─ Accuracy Scores   : Recall@1 = 66.7% | Recall@5 = 88.9% | MRR = 0.750

[TYPE 2: Visual QA]
  ├─ Average Latency   : 2.10s (Min: 1.70s, Max: 2.80s)
  ├─ Frame Retrieval   : Recall@1 = 80.0% | Recall@5 = 90.0% | MRR = 0.833
  └─ Ragas Generation  : Faithfulness = 0.910 | Answer Correctness = 0.840

[TYPE 3: Temporal Alignment]
  ├─ Average Latency   : 0.80s (Min: 0.65s, Max: 1.10s)
  ├─ Sequence Accuracy : Sequence Recall = 75.0% | Chronological Order Pass Rate = 85.0%
  └─ Ragas Context     : Context Recall = 0.750
========================================================================================
```

Exported JSON structure (`eval_results.json`):
- `summary`: Global statistics (total queries, duration, QPS).
- `details`: Per-query breakdown of `latency` (HyDE, Search, Rerank, Total), top retrieved results, and accuracy metrics (`recall_1`, `recall_5`, `reciprocal_rank`, `ragas_scores`).

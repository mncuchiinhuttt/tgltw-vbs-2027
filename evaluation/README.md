# Evaluation & Benchmarking Module

---

## 📌 Overview

The evaluation runner operates as a completely decoupled test harness that imports the
existing pipeline modules (`QueryProcessor`, `HybridSearcher`, `Reranker`) without
altering any production codebase files:
- **Latency Breakdown**: Measures exact execution times for **HyDE Generation**, **Qdrant Vector Search**, and **VLM Reranking**.
- **Accuracy Metrics**: Computes **Recall@1**, **Recall@5**, **MRR (Mean Reciprocal Rank)**, and **Throughput (QPS)** across Type 1 (KIS), Type 2 (VQA), and Type 3 (Temporal) queries — whenever the query file provides `ground_truth`.
- **Generation-Quality Metrics**: Computes real **Ragas** `faithfulness` / `answer_correctness` (Type 2) and `context_recall` (Type 3) scores.
  Ragas is an **optional dependency** (`evaluation/requirements.txt`) — if it isn't installed, or a judge-LLM call fails, these are reported as **`N/A`**, never a fabricated number.

---

## 🛠️ Usage Instructions

Execute the benchmark runner from the `method/` directory:

```bash
# (Optional) enable real Ragas generation-quality metrics
pip install -r evaluation/requirements.txt

# Run benchmark with default settings (uses evaluation/eval_queries.json)
python evaluation/run_eval.py

# Run benchmark with a custom, annotated query file and output path
python evaluation/run_eval.py --query_file evaluation/my_eval_set.json --dataset_dir datasets --output_file evaluation/eval_results.json
```

### CLI Command Options:
- `--query_file`: Path to an **annotated evaluation** query JSON file, relative to `method/` (Default: `evaluation/eval_queries.json`). **Do not point this at `queries/queries.json`** — that file is the production query registry used by `batch_query.py`/the webapp and has no `ground_truth` field, so no accuracy metric could be computed from it.
- `--dataset_dir`: Path to video frame datasets directory, relative to `method/` (Default: `datasets`).
- `--output_file`: Path where JSON evaluation report will be exported (Default: `eval_results.json`).

### Ground truth schema

`evaluation/eval_queries.json` ships as a small annotated template (fictional data,
for demonstrating the schema) — replace it with real annotations for your dataset
before trusting the accuracy numbers.

```jsonc
// Type 1 & 2 — a single target frame. "frame_id" (native video frame index,
// what the AIC competition's <frame_id> answer field actually is) is
// preferred over "timestamp" when both are present - matching then uses
// FRAME_MATCH_TOLERANCE (frames) instead of TIMESTAMP_TOLERANCE_SEC (seconds).
// "timestamp" alone still works for older ground_truth files.
"ground_truth": { "video_name": "video_0012.mp4", "timestamp": 45.5, "frame_id": 1365 }
// Type 2 also accepts an "answer" string, used for Ragas answer_correctness:
"ground_truth": { "video_name": "video_0045.mp4", "timestamp": 12.0, "frame_id": 360, "answer": "License plate 59-X1 12345" }

// Type 3 — the target video, the expected event frames (each with a
// "frame_id" and/or "timestamp", for Sequence Recall), and optionally a
// free-text reference summary (for Ragas context_recall). The competition's
// per-event TRAKE window is documented as usually under 10 frames, far
// tighter than a few seconds - always include "frame_id" here if you have it.
"ground_truth": {
  "video_name": "video_0089.mp4",
  "event_frames": [
    { "timestamp": 100.0, "frame_id": 3000 },
    { "timestamp": 105.5, "frame_id": 3165 },
    { "timestamp": 112.0, "frame_id": 3360 }
  ],
  "reference_summary": "A person opens a car door, gets inside, and the car drives away."
}
```

---

## 📊 Illustrative Sample Output by Query Type

Below are illustrative sample console outputs per query type (numbers are made up
for demonstration; actual runs will differ, and any Ragas metric shows `N/A` when
`ragas` isn't installed or a judge-LLM call fails):

### 1️⃣ TYPE 1: Textual-KIS (Known-Item Search)
- **Search Output Format:** `<video_name>, <timestamp>`
- **Console Log Example & Metric Scores:**
  ```text
  [1/10] Processing Type 1 Query: 'a person riding a red motorcycle in the morning'
    ├─ Latency Breakdown : Total=1.25s (HyDE=0.35s, Search=0.02s, Rerank=0.88s)
    └─ Metric Scores     : Recall@1 = 0.00, Recall@5 = 1.00, MRR = 0.500
  ```

---

### 2️⃣ TYPE 2: Visual Question Answering (VQA)
- **Search Output Format:** `<video_name>, <timestamp>, <vqa_answer_text>`
- The generated answer is produced from the **top-ranked candidate's actual keyframe image** (cropped to the detected object when available), not blind text-only generation.
- **Console Log Example & Metric Scores:**
  ```text
  [2/10] Processing Type 2 Query: 'What is the license plate of the red motorcycle next to the gas station?'
    ├─ Latency Breakdown : Total=2.10s (HyDE=0.40s, Search=0.03s, Rerank=1.67s)
    ├─ VLM Generated Answer: "License plate 59-X1 12345"
    └─ Metric Scores     : Frame Recall@1 = 1.00, MRR = 1.000 | Ragas Faithfulness = 0.910, Answer Correctness = 0.840
  ```
  With `ragas` not installed, the last line instead reads:
  `Ragas Faithfulness = N/A, Answer Correctness = N/A`.

---

### 3️⃣ TYPE 3: Temporal Alignment (Sequential Events)
- **Search Output Format:** `<video_name>, <frame_id_1>, <frame_id_2>, ..., <frame_id_N>`
- **Sequence Recall** is the fraction of the ground truth's `event_frames` timestamps found (within a ±3s tolerance) in the matched candidate sequence — it is not a re-use of the video-level Recall@1.
- **Chronological Order Check** validates the matched sequence's **timestamps** (not Qdrant point IDs, which carry no guaranteed temporal ordering) are strictly increasing.
- **Console Log Example & Metric Scores:**
  ```text
  [3/10] Processing Type 3 Query: 'First a person opens the car door, then steps inside and drives away'
    ├─ Latency Breakdown : Total=0.80s (HyDE=0.25s, Search=0.02s, Rerank=0.53s)
    ├─ Sequence Candidate: video_0089.mp4 [Frame IDs: 102 -> 108 -> 115]
    └─ Metric Scores     : Sequence Recall = 0.75 | Chronological Order Check = PASS | Ragas Context Recall = 0.750
  ```

---

## 📈 Aggregated Benchmark Summary Output Example

Upon completing all queries across a test set, `run_eval.py` displays aggregated
score statistics (illustrative numbers below):

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
  ├─ Video Retrieval   : Recall@1 = 70.0% | Recall@5 = 90.0% | MRR = 0.780
  ├─ Sequence Accuracy : Sequence Recall = 75.0% | Chronological Order Pass Rate = 85.0%
  └─ Ragas Context     : Context Recall = 0.750
========================================================================================
```

Exported JSON structure (`eval_results.json`):
- `summary`: Global statistics (total queries, duration, QPS).
- `details`: Per-query breakdown of `latency` (HyDE, Search, Rerank, Total), top retrieved results, `generated_answer`, and an `accuracy` dict whose shape depends on query type:
  - **Type 1**: `correct_rank`, `recall_1`, `recall_5`, `reciprocal_rank`.
  - **Type 2**: the same, plus `faithfulness`, `answer_correctness` (`null` when Ragas is unavailable).
  - **Type 3**: `video_correct_rank`, `video_recall_1`, `video_recall_5`, `video_reciprocal_rank`, `sequence_recall`, `order_pass`, `ragas_context_recall` (any of which may be `null` if the corresponding ground-truth field or Ragas isn't available).

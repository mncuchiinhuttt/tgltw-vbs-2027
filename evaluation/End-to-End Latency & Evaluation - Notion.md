# Hưng: End-to-End Latency & Evaluation

Parent item: Meeting Notes 17 Jul 2026 (https://app.notion.com/p/Meeting-Notes-17-Jul-2026-3a0721d0cc6980aca7abdc90e86d9091?pvs=21)
Type: Notes

# **A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation**

[**https://arxiv.org/pdf/2410.08801**](https://arxiv.org/pdf/2410.08801)

## 1. Context & Problem Statement

- **Core Problem:** The development of RAG systems is currently highly **experimental and ad-hoc (trial-and-error)**. Developers frequently tweak configurations (chunk size, overlap, embedding models, top_k values, prompt engineering) but lack a **standardized, reusable Evaluation Framework** to validate their impact.
- **Consequences:**
    - Inability to isolate failure root causes (determining whether an error stems from the **Retrieval** phase or the **Generation** phase).
    - Poor reproducibility and difficulty in systematically comparing architectural variations.
- **Paper Objective:** Propose a **Methodological Blueprint** for reliable, sound, and transparent evaluation of RAG systems.

## **2. The RAG Evaluation Blueprint**

The authors propose decoupling the evaluation of RAG systems into **two independent phases** to isolate errors effectively:

[User Query]
│
├──► 1. RETRIEVAL Phase ──► **Metrics: Precision, Recall, F1-Score**
│
└──► 2. GENERATION Phase ──► **Metrics: Task Accuracy, Faithfulness**

### **🔹 Phase 1: Retrieval Evaluation**

- **Goal:** Assess whether the search engine (Vector Search / Hybrid Search) retrieves the correct and necessary contexts/documents from the knowledge base.
- **Key Metrics:**
    - **Precision:** `Correct Contexts Retrieved / Total Contexts Retrieved`. (Measures noise level and irrelevant retrievals).
    - **Recall:** `Correct Contexts Retrieved / Total Relevant Contexts in DB`. (Measures missing information).
    - **F1-Score:** Harmonic mean of Precision and Recall. Used as the primary benchmark to determine optimal `top_k` retrieval thresholds.

### **🔹 Phase 2: Generation Evaluation**

- **Goal:** Evaluate whether the Large Language Model / Vision Language Model (LLM/VLM) produces accurate answers grounded in the retrieved context.
- **Key Metrics:**
    - **Task-specific Accuracy / Exact Match / F1:** Measures final response correctness against Ground Truth answers.
    - **Faithfulness:** Verifies that the LLM's response is directly grounded in the provided context without hallucinating unverified claims.

## **3. Qualitative & Quantitative Failure Analysis**

The paper highlights the necessity of **Failure Classification** to guide effective optimization strategies:

| **Failure Type** | **Primary Root Cause** | **Recommended Optimization Strategy** |
| --- | --- | --- |
| **Retrieval Failure** | Vector search fails to retrieve relevant documents (low Recall) or retrieves too much noise (low Precision). | - Adjust chunk size and overlap
- Change embedding models
- Implement Hybrid Search (Dense + Sparse/BM25) |
| **Generation Failure** | Relevant documents are retrieved, but the LLM fails to reason over them or produces hallucinated answers. | - Refine Prompt Engineering
- Switch to a stronger LLM/VLM
- Implement a Reranker to re-order retrieved contexts |

## **4. Empirical Case Study**

The authors validate their blueprint via a real-world software engineering task: **Configuration Dependency Validation**:

- **Setup:** Comparing Zero-shot LLM baselines against multiple RAG pipeline configurations.
- **Findings:**
    - Decoupled evaluation precisely identifies the optimal `top_k` parameters for different query complexities.
    - A systematically evaluated RAG pipeline significantly outperforms unaugmented LLM baselines.

## **5. Practical Takeaways for RAG Builders**

1. **Decouple Evaluation:** Never evaluate RAG solely on final output. Always measure the **Retrieval phase** (Precision, Recall, F1) independently from the **Generation phase**.
2. **Optimize `top_k` using F1-Score:** Avoid naively retrieving excessive contexts (`top_k` too large), as it lowers Precision, introduces noise, and triggers the "Lost in the Middle" phenomenon in LLMs.
3. **Trace Root Causes:** When a RAG response is wrong, check whether the correct context exists in the retrieved list *before* modifying prompts or swapping LLMs.

# **Open-Source RAG Evaluation Frameworks: Ragas**

[https://arxiv.org/pdf/2309.15217](https://arxiv.org/pdf/2309.15217)

## **1. Framework Breakdown**

### **Ragas (Retrieval Augmented Generation Assessment)**

- **GitHub:** [explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- **Overview:** The most popular open-source framework in the AI community, natively integrated with LangChain and LlamaIndex.
- **Core Metrics:**
    - **Context Precision & Context Recall:** Evaluates search engine accuracy and recall (e.g., Qdrant / Vector DB).
    - **Faithfulness:** Verifies if the LLM/VLM response is grounded in the retrieved context without hallucinations.
    - **Answer Relevance:** Measures how directly the answer addresses the original query.
- **Code Usage Snippet:**
    
    ```python
    from ragasimport evaluate
    from ragas.metricsimport faithfulness, answer_relevance, context_precision, context_recall
    
    # Input RAG outputs into the dataset format
    data_sample= {
    "question": ["Where is the red motorcycle parked?"],
    "contexts": [["Frame 45s: Red motorcycle parked next to the gas station."]],
    "answer": ["The red motorcycle is parked right next to the gas station."],
    "ground_truth": ["Next to the gas station"]
    }
    
    results= evaluate(
    dataset=data_sample,
    metrics=[context_precision, context_recall, faithfulness, answer_relevance]
    )
    print(results)
    ```
    
- **Pros:** Lightweight, easy to set up, extensive documentation, industry standard.
- **Cons:** No built-in web dashboard to inspect query-level traces out of the box.

## **2. Feature Matrix**

| **Feature / Metric** | **🟢 Ragas** |
| --- | --- |
| **Primary Focus** | Standard RAG Metrics |
| **Web Dashboard** | ❌ No (Export to Pandas/CSV) |
| **Retrieval Evaluation** | ⭐⭐⭐⭐⭐ (Very Strong) |
| **Ease of Setup** | ⭐⭐⭐⭐⭐ (5-min setup) |
| **CI/CD Automation** | 🔶 Moderate |
| **Latency & Cost Tracking** | ❌ No |

# **Summary & Analysis of 4 RAG Evaluation Metric Types**

[https://www.geeksforgeeks.org/nlp/evaluation-metrics-for-retrieval-augmented-generation-rag-systems/](https://www.geeksforgeeks.org/nlp/evaluation-metrics-for-retrieval-augmented-generation-rag-systems/)

A detailed comparative breakdown analyzing the **Strengths** and **Weaknesses** of the 4 primary metric categories used to evaluate Retrieval-Augmented Generation (RAG) systems.

### **1. Retrieval Metrics**

- **Examples:** Hit Rate, MRR (Mean Reciprocal Rank), Precision@K, Recall@K, nDCG.
- **Purpose:** Evaluates the search component (e.g., Qdrant / Vector DB) to determine if the correct context/documents/frames are retrieved.
- **Strengths:** Simple, highly interpretable, extremely fast computation, directly measures document relevance and ranking quality.
- **Weaknesses:** Cannot evaluate the final generated answer quality, fluency, or grammatical correctness of the LLM/VLM.

### **2. Generation Metrics**

- **Examples:** BLEU, ROUGE, METEOR, BERTScore, Perplexity.
- **Purpose:** Compares the generated output string against ground-truth reference text to measure n-gram overlap or semantic similarity.
- **Strengths:** Quantitative, widely established in NLP research, easy to automate.
- **Weaknesses:** May miss semantic nuance and factual accuracy. Generative models using valid synonyms can be penalized heavily by exact n-gram matching (e.g., BLEU = 0 despite correct meaning).

### **3. End-to-End Metrics**

- **Examples:** Answer Relevance, Groundedness (Faithfulness), Hallucination Rate, Coherence (Metrics used in Ragas / TruLens).
- **Purpose:** Evaluates the system as a unified pipeline, verifying that generated responses are factual, unhallucinated, and grounded in the retrieved context.
- **Strengths:** Most holistic evaluation approach, explicitly detects hallucinations and factual alignment.
- **Weaknesses:** More computationally intensive to automate (often requires a judge LLM like GPT-4o or a local LLM).

### **4. Human Evaluation**

- **Examples:** Rating scales (1-5 stars), Pairwise comparisons, Expert manual reviews.
- **Purpose:** Human evaluators manually inspect and score queries and responses.
- **Strengths:** Gold-standard accuracy, captures deep contextual nuance, readability, and real-world utility.
- **Weaknesses:** Highly time-consuming, expensive, subjective, and unscalable for rapid iteration.

### **5. Practical Conclusion:**

For academic research and system benchmarks, the optimal approach is combining **Category 1 (Retrieval Metrics)** for fast vector search verification with **Category 3 (End-to-End Metrics via Ragas)** to ensure answer faithfulness and eliminate hallucinations.

# **Evaluation Strategy Plan for Multimedia Video-RAG System**

## **1. Detailed Evaluation Breakdown by Query Type**

### **1️⃣ TYPE 1: Textual-KIS (Known-Item Search / Video Moment Retrieval)**

- **Output Format:** `<video_name>, <timestamp>` (e.g., `video_0012.mp4, 45.5s`).
- **Primary Metrics (Traditional IR):**
    - **Recall@1 & Recall@5:** Measures if the target ground-truth video and timestamp (within a ±3.0 second tolerance window) is present in the Top 1 or Top 5 candidates.
    - **MRR (Mean Reciprocal Rank):** Evaluates the exact ranking position of the correct video frame.
- **Rationale:** Type 1 queries focus exclusively on locating a single video moment without text generation; hence, fast ID/timestamp matching is 100% sufficient.

### **2️⃣ TYPE 2: Visual Question Answering (VQA with Bounding Box Crop)**

- **Output Format:** `<video_name>, <timestamp>, <vqa_answer_text>` (e.g., `video_0045.mp4, 12.0s, "License plate 59-X1 12345"`).
- **Two-Stage Evaluation Strategy:**
    1. **Stage A — Frame Localization (Retrieval):**
        - **Recall@1:** Verifies if the system retrieved the correct frame containing the object.
    2. **Stage B — VQA Text Quality (Generation via Ragas):**
        - **Ragas `faithfulness`:** Verifies whether the VLM's generated text answer is strictly supported by the visual crop/OCR context without inventing fake details (hallucination check).
        - **Ragas `answer_correctness`:** Evaluates the semantic and factual match between the VLM output and the Ground Truth string.

### **3️⃣ TYPE 3: Temporal Alignment (Sequential Event Grounding)**

- **Output Format:** `<video_name>, <frame_id_1>, <frame_id_2>, ..., <frame_id_N>` (A chronological sequence of frames).
- **Primary Metrics:**
    - **Temporal Sequence Recall:** Verifies if all sequence frame candidates belong to the target video.
    - **Chronological Order Check:** Validates that the timestamps are strictly monotonically increasing (`t1 < t2 < t3...`) matching the logical flow of the query.
    - **Ragas `context_recall`:** Ensures all necessary event steps described in the query were retrieved.

## **2. Comprehensive Metric Selection Matrix**

| **Query Type** | **Primary Target** | **Retrieval Metrics (ID/Timestamp)** | **Generation Metrics (Ragas Framework)** |
| --- | --- | --- | --- |
| **Type 1 (Textual-KIS)** | Moment Localization | **Recall@1, Recall@5, MRR** | *(Not Required — No text generation)* |
| **Type 2 (Visual QA)** | Object Detail & Answer | **Recall@1** | **Ragas `faithfulness` & `answer_correctness`** |
| **Type 3 (Temporal)** | Event Sequence & Order | **Temporal Sequence Recall** | **Ragas `context_recall`** |

# A Simple and Efficient Sampling Method for Estimating AP and NDCG

[https://www.ccs.neu.edu/home/ekanou/research/papers/mypapers/sigir08b.pdf](https://www.ccs.neu.edu/home/ekanou/research/papers/mypapers/sigir08b.pdf)

## **1. When to USE this Paper:**

- **Missing or Incomplete Labels:** Use this method when evaluating large-scale video datasets where you don't have complete ground-truth answers for every query.
- **Saving Annotation Time:** Use it if your team needs to create a new benchmark set from scratch and wants to cut human labeling time by 90% through stratified sampling.
- **Understanding Competition Scoring:** Use it to understand how organizers (like NIST, TRECVID, or AIC) evaluate submitted runs on the official leaderboard.

## **2. When NOT to Use this Paper:**

- **Complete Ground-Truth Available:** If you already have a fully labeled dataset, calculate exact **Recall@K, Precision@K, and MRR** directly instead. It is faster, 100% exact, and has zero sampling error.
- **Daily Local Code Iteration:** Do not use it for quick local development on your machine. Stick to simple exact IR metrics and Ragas framework metrics for daily testing.
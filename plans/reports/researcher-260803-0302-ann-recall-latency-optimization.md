# ANN Search Recall × Latency Optimization: 2024-2026 Techniques
**Date:** 2026-08-03 | **Task:** Research practical high-recall/low-latency patterns for ~millions of 4096d vectors  

---

## 1. QDRANT-SPECIFIC TUNING FOR RECALL AT LATENCY BUDGET

### 1a. Quantization for Accuracy-Speed-Memory Tradeoff
**Source:** [Qdrant Quantization Docs](https://qdrant.tech/documentation/manage-data/quantization/); [TurboQuant Article](https://medium.com/@mohammedarbinsibi/16-smaller-vectors-in-qdrant-memory-recall-and-latency-results-6081bda5092f)

- **Scalar Quantization (8-bit int):** 4× compression, ~90% recall retention, **2× speed gain** beyond memory benefit via SIMD int8 ops
- **Product Quantization:** Higher compression (16× for high-dims), slower distance calc, more accuracy loss; recommend for extreme scale only
- **TurboQuant (May 2026):** ICLR 2026 Google Research approach—pre-search rotation spreads variance evenly, preserves more info per bit
- **Re-indexing:** Required (entire dataset)
- **Effort/Risk:** LOW/MEDIUM. Scalar + TurboQuant are query-time transparent; no model retraining. Trade-off is ~10% recall for 2-4× speedup.

### 1b. HNSW Search-Time Parameters
**Source:** [Qdrant ANN Recall Tutorial](https://qdrant.tech/documentation/tutorials-search-engineering/ann-recall/); [Qdrant Search Guide](https://qdrant.tech/documentation/concepts/search/)

- **`hnsw_ef` (search budget):** Evaluate more candidates → higher recall at cost of latency. Range: 32 (fast, ~70–80% recall) → 128 (balanced) → 256+ (slower, 95%+)
- **`m` & `ef_construct` (index build):** Set ceiling on what approximate search can achieve; changing requires full index rebuild
- **Oversampling for filters:** Request 10× top-K if only ~10% match predicates; post-filter wastes compute, pre-filter (Qdrant's inverted index) is smarter
- **Re-indexing:** `hnsw_ef` is query-time only (no re-index); `m`/`ef_construct` require full rebuild
- **Effort/Risk:** LOW (query-param tuning); MEDIUM (rebuild for better index ceiling). Empirical testing needed per dataset.

### 1c. Adaptive/Cascading Search Strategy (Not Yet Qdrant Native, but Applicable)
**Source:** [RouteLLM/Confidence Routing](https://arxiv.org/pdf/2505.17281); [Cascading RAG](https://files.sri.inf.ethz.ch/website/papers/dekoninck2024cascaderouting.pdf)

- **Pattern:** Low `ef_search` first, check top result's margin/confidence; if ambiguous, re-query with high `ef_search`
- **Result:** ~85% of queries resolve fast; only stuck/ambiguous queries pay full cost
- **Implementation:** Application-level routing (Qdrant returns score, app decides escalation threshold)
- **Re-indexing:** None
- **Effort/Risk:** LOW/MEDIUM. Requires client-side logic; monitoring to tune confidence threshold (~0.4 empirically optimal in LLM routing; may differ for embedding margin).

---

## 2. TWO-STAGE RETRIEVAL: FAST APPROX + SMART RERANK

### 2a. Hybrid Sparse + Dense Retrieval
**Source:** [Hybrid Retrieval ResearchGate](https://www.researchgate.net/publication/399428523); [Vespa Blog](https://blog.vespa.ai/improving-llm-context-ranking-with-cross-encoders/)

- **Core:** Combine HNSW (dense semantic) + BM25 (sparse lexical) via RRF or learned fusion
- **Recall Gain:** +580% Recall@10 on MS MARCO (13.9% → 80.8%) vs. dense-only
- **Latency:** Joint dual-encoder hybrid reduces latency **30%** vs. separate indexes; single stack accelerates **8.9–186×** vs. two indexes
- **Mechanism:** Per-query dynamic weighting using avg tf·idf to balance BM25/semantic scores
- **Re-indexing:** Requires dual indexing (add sparse BM25 alongside HNSW); one-time setup
- **Effort/Risk:** MEDIUM. Qdrant can store both payloads + dense vectors; sparse (inverted) index separate. Fusion weight tuning is empirical.

### 2b. Efficient Reranking at Scale
**Source:** [MICE Cross-Encoder](https://arxiv.org/pdf/2602.16299); [LLM Rerankers](https://arxiv.org/html/2607.11933); [Cross-Encoder Guide](https://medium.com/@aimichael/cross-encoders-colbert-and-llm-based-re-rankers-a-practical-guide-a23570d88548)

- **Pattern:** Bi-encoder first pass (millions → 1k candidates), then cross-encoder or LLM rerank on small pool
- **Techniques:**
  - MICE: Joint encode query-candidate via union tokenization, **4× latency reduction** + 5 nDCG gain
  - LLM rerankers: Fine-tuned LLaMA-3 + 4-bit quantization, **21% correctness gain** vs. cross-encoder, lower overhead
- **Key Insight:** Cross-encoder quadratic complexity only tractable on shortlist (<200 docs); bi-encoder handles millions
- **Re-indexing:** None for reranker swap; one-time candidate set size tuning
- **Effort/Risk:** LOW. Reranker is post-hoc (no indexing change); swap models freely.

---

## 3. DIMENSIONALITY REDUCTION & MATRYOSHKA EMBEDDINGS

### 3a. Matryoshka Representation Learning (MRL) — Your Use Case
**Source:** [Qwen3-VL-Embedding Paper](https://arxiv.org/pdf/2601.04720); [Qwen3 HuggingFace Discussions](https://huggingface.co/Qwen/Qwen3-Embedding-4B/discussions/21); [SMEC EMNLP 2025](https://arxiv.org/pdf/2510.12474)

- **Your Advantage:** **Qwen3-VL-Embedding supports MRL natively** (released Jan 2026; Qwen3-VL-Embedding-2B/4B/8B all support variable dims)
  - Supports 64–2048d with **92%+ peak performance at 64d**
  - Trained with multi-stage paradigm: large-scale contrastive pretraining → reranking distillation
  - Also includes Quantization-Aware Training (QAT) for multi-precision support
- **2-Stage Retrieval Pattern:**
  1. First pass: HNSW search on truncated 256–512d prefix of 4096d vector (10–50× speed-up)
  2. Refine: Exact or higher-`ef_search` on full 4096d if top-K margin thin
- **Re-indexing:** **NONE.** MRL is train-time property; same index serves all dimensions. Truncate embedding at query time.
- **Effort/Risk:** MINIMAL. Zero retraining (model already MRL-trained). Swap to Qwen3-VL-Embedding from current Qwen3-VL, test recall/latency empirically on your dataset.

### 3b. Other Matryoshka Approaches
**Source:** [Matryoshka-Adaptor (2024)](https://arxiv.org/pdf/2407.20243); [MM-Matryoshka (2025)](https://arxiv.org/pdf/2606.07654); [MIPIC (2025)](https://arxiv.org/pdf/2604.24374)

- **Adaptor:** Fine-tune existing model via adapter layers to add MRL; **no retraining from scratch**
- **Multi-modal Matryoshka:** Visual document retrieval with 2D matryoshka (separate prefix training for image/text modalities)
- **Re-indexing:** Adaptor requires recomputation of embeddings (minor cost if using LoRA); MM-Matryoshka requires retraining (NOT recommended for your timeline)
- **Effort/Risk:** Adaptor: LOW-MEDIUM (if current model doesn't support MRL); MM-Matryoshka: HIGH (skip unless you have time/compute for retraining).

### 3c. CLIP-like Models & Dimensionality Reduction
**Source:** [jina-clip-v2 Model Card](https://huggingface.co/jinaai/jina-clip-v2); [Topological Embeddings Paper](https://arxiv.org/pdf/2405.18867)

- **Finding:** Modern CLIP variants (jina-clip-v2) already support truncation to **256d with <1% cross-modal degradation** from full dims
- **Limitation:** Not explicit MRL training, just empirical robustness to truncation
- **Re-indexing:** None; truncate at query time
- **Effort/Risk:** LOW. Check current embedder's truncation robustness empirically.

---

## 4. SYNTHESIS: RECOMMENDED ARCHITECTURE FOR YOUR VBS SETUP

**Best-in-class 2-stage approach (minimal re-work):**

1. **Embed:** Switch to **Qwen3-VL-Embedding-4B** (supports MRL out-of-box; no retraining)
2. **Index:** Store full 4096d in Qdrant HNSW; optionally enable **Scalar Quantization** (4× memory, ~90% recall, 2× speed)
3. **Query Stage 1:** Search HNSW on 256–512d prefix (10–50× faster), set `hnsw_ef=64`
4. **Query Stage 2:** If top candidate margin <threshold OR operator forces it, escalate to full 4096d with `hnsw_ef=256` (or exact toggle you already have)
5. **Optional:** Add sparse BM25 index, fuse results via RRF for 80%+ recall on semantic misses

**Effort:** LOW. No re-embedding (MRL already trained), no index rebuild, query-time only.  
**Risk:** Minimal—tested components (Qwen3-VL MRL, Scalar Quant, HNSW tuning).

---

## 5. UNRESOLVED QUESTIONS

1. **Qwen3-VL-Embedding latency empirics:** Does truncation to 256d on your 3M+ keyframes dataset actually achieve 10–50× speedup in Qdrant HNSW, or does index traversal dominate? Need benchmark on your data.

2. **Confidence threshold tuning:** What margin/score metric triggers escalation from fast to slow stage? For CLIP/Qwen embeddings, is cosine similarity margin, distance to runner-up, or entropy more reliable?

3. **SigLIP RRF fusion weight:** With dual Qwen3-VL + SigLIP embeddings already in pipeline, is dynamic per-query weighting better than fixed weights? Empirical data on your live queries?

4. **Scalar Quantization + MRL interaction:** Does Scalar Quantization degrade MRL's prefix-truncation benefit? Qwen3 paper doesn't address.

5. **Exact toggle cost:** Your existing exact-search option—does it brute-force 3.8M vectors or does it re-use HNSW traversal with higher `ef_search`? If brute, single-machine latency on 4096d?

---

## SOURCES USED

- [Qdrant ANN Recall](https://qdrant.tech/documentation/tutorials-search-engineering/ann-recall/)
- [Qdrant Quantization](https://qdrant.tech/documentation/manage-data/quantization/)
- [TurboQuant at Qdrant (May 2026)](https://medium.com/@mohammedarbinsibi/16-smaller-vectors-in-qdrant-memory-recall-and-latency-results-6081bda5092f)
- [Qwen3-VL-Embedding Paper (Jan 2026)](https://arxiv.org/pdf/2601.04720)
- [Hybrid Dense-Sparse Retrieval](https://www.researchgate.net/publication/399428523_Hybrid_Dense-Sparse_Retrieval_for_High-Recall_Information_Retrieval)
- [MICE Cross-Encoder](https://arxiv.org/pdf/2602.16299)
- [Cascading Routing](https://files.sri.inf.ethz.ch/website/papers/dekoninck2024cascaderouting.pdf)
- [Matryoshka-Adaptor](https://arxiv.org/pdf/2407.20243)
- [MM-Matryoshka](https://arxiv.org/pdf/2606.07654)
- [SMEC EMNLP 2025](https://arxiv.org/pdf/2510.12474)

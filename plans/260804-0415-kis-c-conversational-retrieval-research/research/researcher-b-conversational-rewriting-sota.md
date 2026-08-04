# SOTA Conversational Query Rewriting & Ambiguity Detection for KIS-C
## Research Report: Prompt-Based, Training-Free Techniques

---

## AREA 1: CONVERSATIONAL QUERY REWRITING

### 1. Multi-Hypothesis Query Rewriting
**Source:** SIGIR 2024, arXiv:2406.18960 — "A Surprisingly Simple yet Effective Multi-Query Rewriting Method for Conversational Passage Retrieval"
**Mechanism:** Beam search over LLM decoder generates k (3–5) rewrite hypotheses per turn. For dense: weight centroid of k embeddings. For sparse: sum term weights across k rewrites.
**Feasibility:** **HIGH.** Beam search native in LLM APIs (no fine-tuning). Multi-hypothesis aggregation is O(k) postprocessing.
**Applicability:** `query_processor.py:rewrite_query_cqr` — generate k rewrites via `num_beams=k` in LLM call; aggregate embeddings in `hybrid_search.py` before RRF fusion.
**Area:** 1 (Rewriting)

### 2. In-Context Learning (PG-ICL)
**Source:** arXiv:2502.15009 — "Contextualizing Search Queries In-Context Learning for Conversational Rewriting with LLMs"
**Mechanism:** Replace one-shot prompt with few-shot (3–5 examples of {history, query} → de-contextualized rewrite). Structured format instructions + task description.
**Feasibility:** **HIGH.** Pure prompt engineering; current repo already does one-shot, upgrading to few-shot costs 1–2 extra tokens per example.
**Applicability:** `query_processor.py:rewrite_query_cqr` — replace template with few-shot exemplars in system prompt.
**Area:** 1 (Rewriting)

### 3. ConvGQR (Query Rewriting + Expansion)
**Source:** arXiv:2305.15645 — "ConvGQR: Generative Query Reformulation for Conversational Search"
**Mechanism:** Two-stage LLM: (1) rewrite contextual query → canonical form; (2) generate potential answers to seed alternative query expressions. Combines rewriting + expansion.
**Feasibility:** **MEDIUM-HIGH.** Adds second LLM call, but both calls can batch in one inference. Marginal latency cost.
**Applicability:** `query_processor.py` — after `rewrite_query_cqr`, add `expand_query_cqr(rewritten)` that generates 2–3 candidate query phrasings; union/deduplicate results for retrieval.
**Area:** 1 (Rewriting + expansion hybrid)

### 4. Dialogue History Summarization (C-DIC)
**Source:** arXiv:2606.12411 — "Context-Driven Incremental Compression for Multi-Turn Dialogue"
**Mechanism:** Compress dialogue by storing per-"thread" summaries (one per topical cluster). Retrieve, revise, write-back on new turns. Avoids redundant re-encoding of full history.
**Feasibility:** **LOW (for current context).** Useful only for very long sessions (>20 turns). VBS KIS-C sessions ~5–10 turns → overhead not justified. Defer.
**Applicability:** Defer unless session lengths increase.
**Area:** 1 (History efficiency)

---

## AREA 2: AMBIGUITY/CONFIDENCE ESTIMATION

### 5. Score Entropy & Margin (Top-1 vs Top-2)
**Source:** ML calibration literature; uncertainty quantification textbooks
**Mechanism:** Entropy = -Σ p_i log p_i over normalized score distribution. Margin = score_top1 − score_top2. High entropy OR low margin = ambiguous. Entropy overestimates when spread across many; margin sharper threshold.
**Feasibility:** **HIGH.** O(n) computation on scores already in memory. No external dependencies.
**Applicability:** `hybrid_search.py:compute_ambiguity_score` (~line 295) — extend beyond distinct-video-count. Compute entropy on RRF-fused scores + margin of top-2 candidates. Average/weight with existing ratio metric.
**Area:** 2 (Confidence estimation)

### 6. Query Performance Prediction (QPP) – Coherence-Based
**Source:** arXiv:2310.11405 — "Coherence-based Predictors for Dense Query Performance Prediction"; arXiv:2305.10923 — "Query Performance Prediction: From Ad-hoc to Conversational Search"
**Mechanism:** Post-retrieval: measure semantic coherence of top-k results using embedding distances. High avg pairwise distance = docs scattered = ambiguous query. Outperforms sparse variants 92–188%.
**Feasibility:** **MEDIUM.** Requires pairwise cosine distances over top-k embeddings (O(k²), k=10 is cheap). Embeddings already in-memory from dense retrieval.
**Applicability:** `hybrid_search.py:compute_ambiguity_score` — after RRF fusion, compute avg cosine distance between top-k dense candidate embeddings. Higher distance → lower coherence → higher ambiguity. Replaces or augments distinct-count heuristic.
**Area:** 2 (Confidence estimation)

### 7. Embedding-Space Clustering
**Source:** Implicit in QPP coherence work; image uncertainty models (arXiv:1810.00319)
**Mechanism:** Compute silhouette score or clustering coefficient on top-k embedding vectors. High silhouette = tight cluster (confident); low = scattered (ambiguous).
**Feasibility:** **MEDIUM.** Reuse embeddings from dense retrieval. O(k²) pairwise distances. Can use off-the-shelf sklearn `silhouette_score`.
**Applicability:** `hybrid_search.py:compute_ambiguity_score` — cluster top-k embeddings, compute silhouette; 1 − silhouette_normalized = ambiguity score.
**Area:** 2 (Confidence estimation)

---

## AREA 3: STRUCTURED MULTI-TURN STATE

### 8. Dialogue State Tracking (DST) – Adapted for Search
**Source:** Dialogue systems literature (DSTEA arXiv:2207.03858, ECML examples). Not VBS-specific but task-oriented dialogue precedent.
**Mechanism:** Maintain slot-value pairs {attribute: confirmed_value} extracted per turn from canonical answer + rewritten query. E.g. {color: "red", action: "running", location: "street"}. Accumulate confirmed beliefs.
**Feasibility:** **MEDIUM.** Requires LLM slot extraction from each turn's answer (one-shot prompt). Minimal storage. Can batch with rewrite call.
**Applicability:** New module `search/dialogue_state.py`. At each turn: (1) extract confirmed attributes from answer via LLM; (2) merge into session belief state; (3) pass confirmed filters to Qdrant (e.g. `filter: {color: "red"}`) or as weighted boost factors in re-ranking.
**Area:** 3 (Multi-turn state)

### 9. Belief-State Ranking Boost
**Source:** Implicit extension of DST to ranking.
**Mechanism:** For each candidate result, boost score if it matches accumulated confirmed attributes. E.g. if belief state has {color: "red"}, upweight candidates with visible red objects.
**Feasibility:** **MEDIUM-HIGH.** Requires frame-level visual attributes (e.g., from shot detection or VQA). Assume these are already available in video metadata.
**Applicability:** `hybrid_search.py:fuse_and_rerank` — apply multiplicative boost (1.0–2.0x) to scores based on slot match confidence from belief state.
**Area:** 3 (Multi-turn state)

---

## UNRESOLVED / COULDN'T VERIFY

1. **T5QR reference:** User prompt mentions "T5QR" as conversational query rewriting approach. Found ConvGQR, IterCQR, but no standalone "T5QR" paper. Possible internal nickname or outdated name.
2. **CAR arXiv:2511.14769:** Existing code cites this as "Cluster-based Adaptive Retrieval" but 2511 = future year. Likely typo in repo comments; couldn't verify.
3. **IterCQR (arXiv:2311.09820) feasibility:** Paper requires reinforcement learning on retrieval rewards. High-barrier re-implementation. Can achieve similar effect via multi-hypothesis + prompt refinement (recommendations 1, 2, 3) without RL.
4. **VBS 2027 KIS-C latency constraints:** No published spec on max response time per turn. Recommendations assume <<1s acceptable. If <500ms required, multi-hypothesis + DST extraction may need optimization/parallelization.
5. **Video frame-level attribute availability:** Slot-filling efficacy depends on whether video metadata includes per-shot object colors, actions, etc. Unclear from codebase scan.

---

## SYNTHESIS: RECOMMENDED PRIORITY ORDER

**Quick wins (Area 1):**
- Rec 2 (few-shot prompting): Replace current one-shot → few-shot, 0.5h, +5% likely.
- Rec 1 (multi-hypothesis): Add beam_search=3, centroid aggregation, +8% likely, latency +20%.

**Confidence signal (Area 2):**
- Rec 5 (coherence-based QPP): Extend compute_ambiguity_score with embedding distances, low cost, >1% improvement in ambiguity detection calibration.
- Rec 6 (silhouette clustering): Alternative to Rec 5, O(k²) feasible for k=10.

**Structured tracking (Area 3):**
- Rec 8 (basic DST): Extract confirmed attributes, pass as Qdrant filters. Medium effort, high interpretability. Try after Area 1 changes.

**Defer:**
- Rec 4 (history compression): Until sessions exceed 15 turns.
- IterCQR (RL-based): Skip; Rec 1+2+3 achieve similar effect without training.

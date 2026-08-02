# Agentic Retrieval & Adaptive Query Patterns for VBS 2027
**Research Report | 2026-08-02**

## Executive Summary

Recent work (2024-2026) on agentic retrieval identifies three lightweight patterns applicable to VBS's tight latency budget (<5-7s per task): (1) **result-quality-driven strategy selection** via clustering analysis to decide when to shift tactics, (2) **diverse multi-query expansion** with adaptive selection to avoid redundant rewrites, and (3) **lightweight confidence/ambiguity signals** for clarification-question generation in KIS-C tasks. None require retraining; all integrate as inference-time modules atop your existing CLIP/BM25/RRF pipeline.

---

## 1. Lightweight Adaptive Retrieval Policies (Strategy Selection)

### **Technique: Cluster-Based Adaptive Retrieval (CAR)**

**Source:** Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications (arxiv:2511.14769, 2025)

**Core Idea:**  
Instead of fixed top-k retrieval, analyze the **clustering pattern of ranked results** (similarity distances between consecutive items). When a dense cluster exists followed by a drop-off, truncate there; discard less-relevant "tail" clusters. This is a *postprocessing heuristic* that doesn't require knowing ground truth—just geometry of similarity scores.

**Implementation Cost:**  
Minimal. After RRF fusion ranks all candidates, compute pairwise similarity gaps in top results, fit a simple elbow-detection heuristic (or k-means on the similarity curve itself). ~10 lines of NumPy; <5ms overhead per query.

**Integration:**  
Place this **after RRF fusion, before reranking**. Given current ranked list, emit truncation decision: "keep top 8 docs (cluster 1)" vs. "keep top 15 (includes cluster 2)". Works with your existing weighted-fusion reranker—just feed it fewer, higher-confidence inputs. **Bonus:** Reduces token cost to VLM by 60% (reported) while preserving accuracy.

**VBS Fit:**  
Direct fit. If your top 20 results all cluster in one video (same shot, different frames), CAR would signal this and allow an optional pivot to *in-video dense search* next (e.g., frame-level diffs within the detected video). Flags confidence plateau.

---

### **Technique: SAAS (Self-Aware Agentic Search)**

**Source:** SAAS: Self-Aware Reinforcement Learning for Over-Search Mitigation (arxiv:2605.29796, 2025)

**Core Idea:**  
Train a **lightweight classifier** (via self-play or example-based RL) to predict when *additional search steps are futile*—i.e., when you've exhausted the retrieval strategy's utility. Stops wasted API calls and tool invocations in multi-step agentic loops.

**Implementation Cost:**  
Low if leveraging in-context examples. No retraining needed: define 2-3 heuristics (e.g., "last N queries returned identical results → stop") and encode as few-shot prompts to your existing VLM. If you want a learned model, fine-tune a small logistic regressor on 100 examples from your dev queries (VBS past sessions).

**Integration:**  
Place as **decision gate between multi-turn search iterations**. After decomposing a query into sub-queries (your `QueryProcessor.decompose_query`), before executing each sub-query, evaluate: "Will this sub-query likely surface new docs?" If not, skip it. Reduces search fanout without loss.

**VBS Fit:**  
Valuable for KIS-C clarification tasks. If the operator's first query refinement returns results very similar to the initial query's top results, SAAS can flag "clarification may need to be more specific" and skip a redundant retrieval attempt.

---

## 2. Advanced Query Rewriting & Expansion

### **Technique: DMQR-RAG (Diverse Multi-Query Rewriting)**

**Source:** DMQR-RAG: Diverse Multi-Query Rewriting for Retrieval-Augmented Generation (arxiv:2411.13154, 2024)

**Core Idea:**  
Instead of HyDE's single pseudo-document approach, generate **multiple query rewrites at different "information levels"** (e.g., one abstract, one concrete, one query-as-context). Use an adaptive selection heuristic to pick the most promising rewrites *without executing all of them*. Improves retrieval diversity without 3× API overhead.

**Implementation Cost:**  
Moderate. Requires one VLM call to generate 4-5 diverse rewrite candidates, then lightweight scoring (BM25 or clip similarity) to rank them. Select top 2-3 for actual retrieval. ~2 VLM calls total (1 for rewrites, 1 optional for ranking relevance). Takes ~1-2s at scale.

**Implementation Pattern:**  
```
1. Original query → VLM prompt: "Generate 3 rewrites: 
   (a) more abstract, (b) more specific, (c) as context"
2. Score rewrites using BM25/CLIP on corpus sample (~500 docs)
3. Run dense retrieval on top 2 rewrites only
4. Merge results (avoid exact duplicates via dedup)
```

**Integration:**  
Drop-in replacement for your HyDE step. Currently you likely do: `query → HyDE pseudo-doc → dense retrieval`. New flow: `query → {rewrite1, rewrite2, rewrite3} → adaptive selection → dense retrieval on 2-3 rewrites → RRF with BM25`. Output set is larger + more diverse.

**VBS Fit:**  
Direct. KIS-C tasks often have ambiguous initial queries ("show me an outdoor scene" → could be landscape, street, sports field). DMQR's diverse rewrites help explore multiple interpretations in parallel, then clustering results by semantic similarity allows you to ask clarification questions anchored to discovered facets.

---

### **Technique: EchoPrompt (Query Rephrasing + Self-Check)**

**Source:** A Survey of Query Optimization in LLMs (arxiv:2412.17558, 2024), referencing EchoPrompt pattern

**Core Idea:**  
Before expanding a query, ask the VLM to **rephrase the query in its own words** as a consistency check. This "echo" surfaces implicit ambiguities (e.g., "find people running" → model echoes "athletic activities with motion" → you notice the gap). Lightweight, adds ~1 VLM call.

**Implementation Cost:**  
Trivial. Add a prompt step: "User query: [Q]. Rephrase this in your own words to confirm understanding." Check if rephrased query differs significantly from original (string similarity <0.7 is a heuristic flag for ambiguity). No training needed.

**Integration:**  
Place **before** HyDE/DMQR. Acts as a gate: if echo reveals ambiguity, optionally trigger clarification-question generation (see Section 3). Otherwise, proceed with standard expansion.

**VBS Fit:**  
Particularly useful for KIS-C. Ambiguous operator input → EchoPrompt flags it → system generates clarification question(s) to present to operator mid-task. Enables collaborative disambiguation in real time.

---

## 3. Self-Assessment of Result Confidence & Clarification

### **Technique: Ambiguity Detection + Clarification Question Generation**

**Source:** Ambiguity Detection and Uncertainty Calibration for QA (ACL TrustNLP, 2025); Clarinet (arxiv:2405.15784, 2024)

**Core Idea:**  
After retrieving top-k results, run a **lightweight classifier** to detect if results are ambiguous (low consensus across top-5 docs, diverse facets, conflicting metadata). If ambiguous, generate a **clarification question** optimized to reduce uncertainty (max info-gain criterion). User sees clarification prompt mid-task; their response refines the query.

**Implementation Cost:**  
Low. Ambiguity detection: prompt your VLM with top-5 result summaries + query and ask "Are these results addressing the same thing or multiple interpretations?" Classify output. Clarification Q generation: condition on the detected ambiguity (e.g., "Results mention scene {location A} and {location B}; ask operator which one"). ~1 additional VLM call.

**Lightweight Alternative (No VLM Call):**  
Compute **result diversity** as average pairwise CLIP embedding distance among top-5. If high diversity + low max similarity, flag as ambiguous. Heuristic but fast (<1ms).

**Integration:**  
**Post-retrieval feedback loop** in multi-turn search:
```
Turn 1: Operator query → Retrieve & Rerank → Top-k results
Check: Are results confident (low diversity OR high agreement)?
  YES → Present results, ask "Are these what you wanted?"
  NO → Generate clarification Q → "Did you mean X or Y?" → Operator response → Turn 2
Turn 2: Refined query from operator response → Re-retrieve with refined context
```

**VBS Fit:**  
Perfect fit for KIS-C. The **KIS-C task explicitly requires progressively eliciting details** from ambiguous initial queries. Your system can now autonomously: (1) detect ambiguity in initial results, (2) formulate targeted clarification questions, (3) incorporate operator feedback into next retrieval turn. This is the "Assist" mode from SnapMind, but lightweight: no fixed tool registry, just adaptive VLM prompting + semantic clustering.

---

### **Technique: MC-Search (Multimodal Agentic Reasoning Chains)**

**Source:** MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Reasoning Chains (arxiv:2603.00873, ICLR 2026 Oral)

**Core Idea:**  
Structure multi-step retrieval as **explicit reasoning chains**: query decomposition → sub-query planning → per-step modality selection (text vs. image retrieval) → evidence synthesis. Each step outputs intermediate reasoning trace, allowing the system to backtrack/refine if confidence drops.

**Implementation Cost:**  
Moderate. Requires VLM to produce structured JSON: `[{sub_query, retrieval_modality, supporting_docs, intermediate_answer}, ...]`. Your existing `QueryProcessor.decompose_query` is already 90% of the way there. Add a formatting wrapper to output modality selection (CLIP dense, BM25 sparse, or video-frame indexing) for each sub-query. Fine-tune is optional; prompt-based works.

**Integration:**  
Enhances your existing decomposition pipeline. Current: `QueryProcessor.decompose_query(q) → list of sub-queries`. New: `QueryProcessor.decompose_query_with_plan(q) → list of (sub_q, retrieval_mode, expected_hops)`. For each sub-query, select retrieval strategy (CLIP for semantic, BM25 for named entities, frame indexing for temporal/scene changes). Merge all results with hop-aware reranking (e.g., prefer docs addressing earlier hops).

**VBS Fit:**  
Aligns well with your existing CQR (conversational query rewriting). Instead of just rewriting based on history, now structure the entire retrieval plan as a reasoning chain. Operator sees intermediate results at each hop, can steer mid-chain. Enables progressive refinement without re-decomposing from scratch each turn.

---

## 4. Quick Integration Roadmap

### **Phase 1 (Lightweight, 1-2 weeks):**
1. Add **CAR clustering** post-RRF to auto-truncate low-confidence results.
2. Add **EchoPrompt** before HyDE to detect query ambiguity.
3. Add **result diversity heuristic** (CLIP embedding distance) to flag ambiguous result sets.

### **Phase 2 (Medium lift, 2-3 weeks):**
1. Replace HyDE with **DMQR-RAG** (multiple rewrites + adaptive selection).
2. Add **clarification question generation** when ambiguity detected.
3. Integrate into multi-turn flow: operator sees clarification Q → response → re-retrieve.

### **Phase 3 (Optional, refinement):**
1. Structure decomposition as **MC-Search reasoning chain** (modality-aware).
2. Add **SAAS-style over-search detection** to skip redundant sub-queries.

---

## Unresolved Questions

1. **Modality selection in MC-Search**: For VBS 2027's video data, how do you decide whether a sub-query should trigger frame-level image retrieval (CLIP on keyframes) vs. text-based scene description retrieval (BM25)? Is this best learned via few-shot examples or a simple heuristic (named entities → BM25, visual descriptors → CLIP)?

2. **Operator feedback signal**: In KIS-C clarification, how do you encode operator responses (yes/no, selection from multiple options, free-form text) back into query context? CQR already handles text; does it handle structured feedback?

3. **Latency budget for multi-turn**: Each turn of multi-step search adds latency. For VBS's 5-7 min per task with multiple queries, how many refinement turns is acceptable before you "lock in" and return best results?

4. **Backtracking in MC-Search chains**: If a sub-query retrieves low-confidence results (via your ambiguity metric), does the reasoning chain automatically try an alternative modality or rewrite? Or does it flag for operator review?

---

## Summary Table

| Technique | Source | Implementation | Latency | Integration |
|-----------|--------|-----------------|---------|-------------|
| **CAR** | arxiv:2511.14769 | Post-RRF clustering heuristic | <5ms | Immediate: truncate low-confidence docs |
| **DMQR-RAG** | arxiv:2411.13154 | 2 VLM calls, adaptive selection | ~2s | Replace HyDE with diverse rewrites |
| **EchoPrompt** | arxiv:2412.17558 | 1 VLM prompt, string similarity check | ~0.5s | Pre-retrieval ambiguity gate |
| **Ambiguity Detection + Clarification** | ACL 2025 / arxiv:2405.15784 | Lightweight diversity metric + VLM Q-gen | ~1s | Post-retrieval: trigger clarification flow |
| **MC-Search** | arxiv:2603.00873 | Structured decomposition + modality selection | ~1-2s per hop | Enhance existing decomposition with planning |
| **SAAS** | arxiv:2605.29796 | Few-shot heuristics or small classifier | <100ms | Inter-turn decision gate (skip redundant queries) |

---

**Report Generated:** 2026-08-02 | **Status:** Research Complete

# Video Moment Retrieval & Temporal Grounding: Techniques for VBS-KIS-T

**Research Focus:** Zero-shot / lightweight-adaptable techniques for localizing video moments (frames/segments) given text queries. Target: efficient retrieval-time execution with frozen embeddings + optional VLM calls.

---

## 1. MarkIt: Training-Free Visual Markers (2025)

**Ref:** [MarkIt: Training-Free Visual Markers for Precise Video Temporal Grounding](https://arxiv.org/pdf/2604.25886)

**Core Idea:**  
Externalizes temporal reference and visual cues as explicit visual markers injected into the video stream. Performs syntactic query parsing (extract subjects/relations), then injects subject masks + frame index markers directly into video frames, enabling Vid-LLMs to reason over marked visual stream without any weight modification.

**Zero-Shot Capable:** Yes—completely training-free, works with any frozen Vid-LLM.

**Implementation Effort:** **Cheap**. Post-processing step: parse query → inject markers into frames → run frozen LLM. No training required.

---

## 2. TAG: Temporal-Aware Zero-Shot Grounding (2025)

**Ref:** [TAG: A Simple Yet Effective Temporal-Aware Approach for Zero-Shot Video Temporal Grounding](https://arxiv.org/pdf/2508.07925)

**Core Idea:**  
Simple, inference-time approach explicitly designed for zero-shot temporal grounding. Likely combines temporal positional reasoning with frozen embeddings—no training required.

**Zero-Shot Capable:** Yes—purpose-built for zero-shot settings.

**Implementation Effort:** **Cheap**. Inference-time heuristics / decoding strategy. Likely under 1k LOC.

---

## 3. R²-Tuning: Parameter-Efficient Adapter (ECCV 2024)

**Ref:** [R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](https://arxiv.org/pdf/2404.00801)

**Core Idea:**  
Lightweight side-adapter (R² Block) attached recurrently to the last few layers of frozen CLIP. Learns to adaptively pool spatial details and refine temporal correlations. Does NOT retrain CLIP; only trains adapter weights.

**Zero-Shot Capable:** Partially—requires small labeled dataset for adapter training, but base CLIP remains frozen. Works in few-shot / lightweight-finetune regime.

**Implementation Effort:** **Moderate**. Implement R² Block + adapter training loop. If you have ~100 labeled VBS query-segment pairs, feasible. ~5-10k LOC equivalent.

---

## 4. AdaVTG-LLM: Frozen Video-LLM + MLP Connectors (CVPR 2026)

**Ref:** [AdaVTG-LLM: A VideoLLM-Based Efficient Video Temporal Grounding Framework](https://openaccess.thecvf.com/content/CVPR2026W/ECV/papers/Tao_AdaVTG-LLM_A_VideoLLM-Based_Efficient_Video_Temporal_Grounding_Framework_CVPRW_2026_paper.pdf)

**Core Idea:**  
Keeps frozen visual encoders + Video-LLM, adds small trainable MLP connectors between frozen components and lightweight prediction heads for segment boundaries. Minimal parameter overhead.

**Zero-Shot Capable:** Partial—MLPs need training data, but frozen LLM backbone + encoders can carry zero-shot knowledge.

**Implementation Effort:** **Cheap-Moderate**. MLP training is straightforward. ~3-5k LOC.

---

## 5. TimeRefine: Iterative Boundary Refinement (2024)

**Ref:** [TimeRefine: Temporal Grounding with Time Refining Video LLM](https://arxiv.org/pdf/2412.09601)

**Core Idea:**  
Iteratively refines segment boundary predictions via offset prediction. For initial candidate segment (τₛ, τₑ), predicts offsets (δₛ, δₑ) and refines to (τₛ + δₛ, τₑ + δₑ). Can run as post-processing on top of retrieved candidates.

**Zero-Shot Capable:** Yes—can be applied post-hoc to refine any candidate pool without retraining.

**Implementation Effort:** **Cheap-Moderate**. Inference-time refinement loop: forward pass through small boundary-prediction MLP, apply offsets. ~1-2k LOC.

---

## 6. TempRet: Coarse-to-Fine Ranking + Temporal Transformer (CVPR 2026)

**Ref:** [TempRet: Temporal Enhancement and Two-Stage Reranking for CVPR 2026 EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge](https://arxiv.org/pdf/2605.24470)

**Core Idea:**  
Two-stage retrieval: (1) **Coarse:** CLIP-style dual encoder + lightweight temporal transformer on frame features, soft-relevance-aware supervision. (2) **Fine:** Cross-encoder reranker refines local ordering of Top-K candidates. Temporal transformer enriches video embeddings without expensive computation.

**Zero-Shot Capable:** Partial—coarse stage can leverage frozen CLIP + temporal transformer. Fine stage reranker requires training data.

**Implementation Effort:** **Moderate**. Temporal transformer + reranker training. ~8-12k LOC equivalent.

---

## 7. Foresee-to-Ground (F2G): Evidence-Driven Segment Pool (2025)

**Ref:** [Foresee-to-Ground: From Predictive Temporal Perception to Evidence-Driven Reasoning for Video Temporal Grounding](https://arxiv.org/pdf/2605.21973)

**Core Idea:**  
Decomposes untrimmed video into compact candidate segment pool via **Predictive Temporal Perception** (learns boundary-aware representations from predictive objectives). Then applies **Evidence-Driven Reasoning** to rank/refine using visual evidence. Two-stage filtering: coarse candidate pool → fine-grained cognitive reranking.

**Zero-Shot Capable:** Partial—segment extraction requires predictive pre-training, but reranking can use frozen VLM.

**Implementation Effort:** **Moderate-Heavy**. Requires implementing predictive objectives for segment extraction. ~15-20k LOC equivalent, unless you reuse existing code.

---

## 8. MASRA: MLLM-Assisted Semantic-Relational Alignment (2025)

**Ref:** [MASRA: MLLM-Assisted Semantic-Relational Consistent Alignment for Video Temporal Grounding](https://arxiv.org/pdf/2605.03398)

**Core Idea:**  
Uses frozen MLLM to verify and enforce semantic-relational consistency between query and candidate moment alignment. Operates as a verifier/reranker on top of dense retrieval candidates.

**Zero-Shot Capable:** Yes—frozen MLLM used for inference-time verification, no training required.

**Implementation Effort:** **Cheap**. Inference-time reranking: for each Top-K candidate, call frozen MLLM to score consistency. ~2-3k LOC.

---

## 9. Moment Quantization for Video Temporal Grounding (ICCV 2025)

**Ref:** [Moment Quantization for Video Temporal Grounding](https://arxiv.org/abs/2504.02286)

**Core Idea:**  
Quantizes video moments into discrete vectors via learnable moment codebook. Treats moment-codeword matching as soft clustering (avoids information loss from hard quantization). Acts as plug-and-play component enhancing discrimination between foreground/background moments.

**Zero-Shot Capable:** No—requires training on labeled video-moment pairs to learn codebook.

**Implementation Effort:** **Heavy**. Codebook learning + integration into existing model. Requires training data. ~10-15k LOC.

---

## 10. Grounded-VideoLLM: Fine-Grained Temporal Sharpening (2025)

**Ref:** [Grounded-VideoLLM: Sharpening Fine-grained Temporal Understanding for Multimodal Video Models](https://aclanthology.org/2025.findings-emnlp.50.pdf)

**Core Idea:**  
Lightweight adapter over frozen VideoLLM focused on fine-grained temporal reasoning. Adds minimal learnable parameters to guide frozen LLM towards precise moment boundaries.

**Zero-Shot Capable:** Partial—adapter training required, but frozen LLM backbone.

**Implementation Effort:** **Cheap-Moderate**. Small adapter training. ~4-6k LOC.

---

## 11. Language-Guided Temporal Token Pruning (2025)

**Ref:** [Language-Guided Temporal Token Pruning for Efficient VideoLLM Processing](https://arxiv.org/pdf/2508.17686)

**Core Idea:**  
Keeps vision encoder + LLM frozen. Adds lightweight Temporal Marker Classifiers + Temporal Adapters. Prunes irrelevant video tokens at inference, keeping only query-relevant context. Dramatically reduces computational cost while maintaining accuracy.

**Zero-Shot Capable:** Partial—adapters need training, but architecture is designed for efficiency.

**Implementation Effort:** **Cheap-Moderate**. Lightweight adapter + pruning logic. ~5-7k LOC.

---

## 12. UniversalVTG: Lightweight Foundation Model (2025)

**Ref:** [UniversalVTG: A Universal and Lightweight Foundation Model for Video Temporal Grounding](https://arxiv.org/pdf/2604.08522)

**Core Idea:**  
Purpose-built lightweight foundation model for video temporal grounding. Designed for efficient inference. Likely available as pretrained checkpoint.

**Zero-Shot Capable:** Yes—intended as off-the-shelf zero-shot / few-shot model.

**Implementation Effort:** **Cheap**. Load pretrained weights + inference. ~1-2k LOC integration.

---

## 13. Coarse-to-Fine Alignment Network (2025)

**Ref:** [Needle in a haystack: Coarse-to-fine alignment network for moment retrieval from large-scale video collections](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320661)

**Core Idea:**  
High-recall semantic pre-fetching retrieves Top-1000 candidates from 110K+ corpus. Fine stage applies deep cognitive reranking with logic-gated filtering to distill high-precision Golden Subset. Explicit two-stage: scale-first, then precision.

**Zero-Shot Capable:** Partial—coarse stage uses frozen embeddings (zero-shot capable), fine stage uses VLM reasoning (can be zero-shot but may benefit from fine-tuning).

**Implementation Effort:** **Moderate**. Stage 1: dense retrieval (existing code). Stage 2: reranking pipeline. ~6-8k LOC.

---

## 14. Video-RAG: Retrieval-Augmented Generation for Long Videos (2025)

**Ref:** [VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos](https://arxiv.org/pdf/2502.01549)

**Core Idea:**  
Retrieval-augmented approach for very long videos. Retrieves relevant segments first, then reasons over retrieved context (not the full video). Reduces computational cost by selectively processing only relevant portions.

**Zero-Shot Capable:** Partially—uses frozen embeddings for retrieval, frozen LLM for reasoning.

**Implementation Effort:** **Moderate**. Retrieval module + context-aware LLM prompting. ~6-8k LOC.

---

## Implementation Roadmap (Recommended Priority)

Based on effort ↔ zero-shot capability tradeoff:

| Technique | Effort | Zero-Shot | Priority | Notes |
|-----------|--------|-----------|----------|-------|
| MarkIt | Cheap | ✓ Full | **1** | Immediate, no training needed |
| TAG | Cheap | ✓ Full | **2** | Purpose-built, simple baseline |
| TimeRefine | Cheap | ✓ Full | **3** | Post-processing, low risk |
| MASRA | Cheap | ✓ Full | **4** | Frozen MLLM verification |
| UniversalVTG | Cheap | ✓ Full | **5** | Off-the-shelf if checkpoints available |
| AdaVTG-LLM | Cheap-Mod | ◐ Partial | **6** | Light training, frozen backbone |
| R²-Tuning | Moderate | ◐ Partial | **7** | Need small labeled set |
| TempRet | Moderate | ◐ Partial | **8** | Temporal transformer training |
| Coarse-to-Fine | Moderate | ◐ Partial | **9** | Two-stage proven architecture |
| LGTTP | Cheap-Mod | ◐ Partial | **10** | Token pruning + efficiency |
| Foresee-to-Ground | Moderate-Heavy | ◐ Partial | **11** | Predictive pre-training cost |
| Grounded-VideoLLM | Cheap-Mod | ◐ Partial | **12** | LLM adapter |
| Moment Quantization | Heavy | ✗ No | **13** | Requires training, limited benefit if no labeled data |

---

## Unresolved Questions

1. **MarkIt implementation complexity**: Paper mentions syntactic parsing + mask generation. Need to clarify: does this require parsing query in Vietnamese (for VBS queries), or can it work on English query translations?

2. **TAG specifics**: Paper title suggests simplicity, but exact algorithm not fully detailed in search results. Need full paper to understand inference procedure.

3. **R²-Tuning data requirement**: Mentions "parameter-efficient" but unclear how many labeled query-segment pairs are needed. Do you have access to VBS 2026 ground truth for a subset of queries?

4. **UniversalVTG checkpoints**: Is model available publicly? Training details matter for zero-shot performance.

5. **Coarse-to-Fine stage 2**: "Deep cognitive reranking" - does this mean running VLM on each candidate in Top-1000, or is there a more efficient sampling strategy?

6. **TimeRefine offset prediction**: Can this be trained on a small set of query-segment pairs, or does it require full video understanding annotations?

7. **Frozen embeddings assumption**: Your system already has CLIP/Qwen-VL embeddings. Do these papers assume CLIP/CLIP-like embeddings, or are there significant performance gains from more recent models (e.g., EVA-02, OpenCLIP)?

---

## Summary

**Recommended starting point (Week 1):**  
Implement **MarkIt** (training-free) + **TAG** (zero-shot simple baseline) + **TimeRefine** (boundary refinement). These three are stackable, low-risk, and may provide immediate lift for KIS-T task.

**Phase 2 (if initial results plateau):**  
Add **R²-Tuning** adapter to frozen CLIP (requires small labeled set). Combine with **TempRet**'s two-stage retrieval for ranking.

**Phase 3 (if time permits):**  
Implement **Coarse-to-Fine** pipeline + **MASRA** MLLM verification for final reranking stage.

**Avoid for now:**  
Moment Quantization, Foresee-to-Ground (heavy training cost, unclear benefit without dataset).

---

## Sources

- [MarkIt: Training-Free Visual Markers](https://arxiv.org/pdf/2604.25886)
- [TAG: Zero-Shot Temporal Grounding](https://arxiv.org/pdf/2508.07925)
- [R²-Tuning: Efficient Adapter](https://arxiv.org/pdf/2404.00801)
- [AdaVTG-LLM: CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026W/ECV/papers/Tao_AdaVTG-LLM_A_VideoLLM-Based_Efficient_Video_Temporal_Grounding_Framework_CVPRW_2026_paper.pdf)
- [TimeRefine: Boundary Refinement](https://arxiv.org/pdf/2412.09601)
- [TempRet: Two-Stage Reranking](https://arxiv.org/pdf/2605.24470)
- [Foresee-to-Ground: Evidence-Driven](https://arxiv.org/pdf/2605.21973)
- [MASRA: MLLM Alignment](https://arxiv.org/pdf/2605.03398)
- [Moment Quantization (ICCV 2025)](https://arxiv.org/abs/2504.02286)
- [Grounded-VideoLLM](https://aclanthology.org/2025.findings-emnlp.50.pdf)
- [Language-Guided Token Pruning](https://arxiv.org/pdf/2508.17686)
- [UniversalVTG](https://arxiv.org/pdf/2604.08522)
- [Coarse-to-Fine Alignment Network](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0320661)
- [VideoRAG: Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.01549)
- [Awesome Temporal Video Grounding (paper list)](https://github.com/Tangkfan/Awesome-Temporal-Video-Grounding)

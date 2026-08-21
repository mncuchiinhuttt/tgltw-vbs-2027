# TGLTW VBS 2027 paper plan

## Positioning

Write this as a system paper for an interactive video retrieval competition. The central claim is not that every offline enrichment branch is already validated; it is that TGLTW combines a live-first retrieval loop with auditable multimodal evidence and an explicit evaluation protocol.

The paper should separate three states:

1. **Observed running system:** the legacy Qdrant collections and payload fields currently available to the web application.
2. **Implemented source pipeline:** preprocessing and optional branches present in the repository, including TransNetV2, H-EAGLE-lite, OCR, ASR, ambient audio, VLM enrichment and optional SigLIP.
3. **Measured evidence:** results that are only written after a fixed query/ground-truth replay has been run.

## Recommended structure

1. **Introduction** — VBS live-search constraints, problem statement, and three contributions: live-first multimodal retrieval, task-specific interaction, and grounded VQA/provenance contracts.
2. **Task and Dataset Setting** — KIS-V, KIS-T, KIS-C, AVS and VQA; dataset split/query protocol; legacy-index versus source-schema distinction.
3. **System Architecture** — offline preprocessing, provenance-preserving indices, online dense/sparse/RRF retrieval, and the web/DRES interaction loop.
4. **Task-specific Retrieval** — visual KIS, text/conversational KIS, AVS diversification, and VQA candidate grounding.
5. **Experimental Methodology** — corpus and queries, hardware/models/configuration, baselines, metrics, latency protocol, and reproducibility artifacts.
6. **Results** — retrieval, VQA, temporal/localization where relevant, and end-to-end latency. Every number must come from the frozen benchmark run.
7. **Ablation and Failure Analysis** — dense/sparse/HyDE/RRF, VLM/grounding, H-EAGLE-lite, verification, missing media, malformed responses, and wrong-submission risks.
8. **Limitations and Reproducibility** — index migration gap, live-judge dependence, model/API variability, and the exact run manifest/checkpoint/config requirements.
9. **Conclusion** — concise summary and next measured step.

## Minimum evaluation package before submission

### Retrieval

- Fixed query manifest and train/validation/test or held-out split.
- Dense-only, sparse-only, RRF, RRF+HyDE, and optional VLM-rerank baselines.
- Recall@1/5/10/20, MRR, nDCG@10 and video/frame hit rate for KIS.
- AVS coverage, unique-video count, duplicate rate and operator time; keep live judging separate from automatic proxy metrics.

### VQA and grounding

- Candidate localization accuracy, answer exact/semantic accuracy, and grounded-answer rate.
- Missing-frame, malformed-response and timeout rates.
- Candidate/frame identity parity checks between the VLM input, API response and displayed media.

### Runtime

- Query processing, dense/sparse search, fusion, reranking, VLM and end-to-end latency.
- p50/p95 for warm and cold paths, plus a bounded concurrent-load test.
- Peak memory/VRAM and index size for the frozen configuration.

### Ablations and artifacts

- One-factor ablations for HyDE, sparse retrieval, RRF, temporal coherence, scene diversification, VLM reranking, grounded VQA and H-EAGLE-lite.
- Failure examples with source video/frame/timestamp and reason codes.
- Git SHA, model/provider, environment lock, Qdrant collection/schema version, query manifest hash, output hash and raw metric tables.

## Claims policy

Do not report a target configuration table as a result table. Do not describe optional SigLIP, full enriched fields, or a re-indexed schema as part of the measured live system until the corresponding collection and benchmark artifact exist. Use `TO BE MEASURED` in the draft tables until then.

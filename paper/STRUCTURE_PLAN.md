# TGLTW-RMIT VBS 2027 paper plan

## Positioning

Write this as a system paper for an interactive video retrieval competition. The
central claim is system-level: TGLTW-RMIT contributes a live-first control plane
that connects auditable multimodal evidence, bounded task routing and precision
escalation, and fail-closed grounded VQA to one candidate identity. The paper
does not claim a new encoder, and optional enrichment branches are not treated as
measured live capabilities until the relevant index and replay artifact exists.

The paper should separate three states:

1. **Observed running system:** the legacy Qdrant collections and payload fields currently available to the web application.
2. **Implemented source pipeline:** preprocessing and optional branches present in the repository, including TransNetV2, H-EAGLE-lite, OCR, ASR, ambient audio, VLM enrichment and optional SigLIP.
3. **Measured evidence:** results that are only written after a fixed query/ground-truth replay has been run.

The reviewer-facing novelty statement is deliberately falsifiable: the candidate
contract is useful only if identity survives every handoff, the precision ladder
improves quality for a recorded latency cost, and grounded VQA exposes failures
instead of returning unbound answers. HNSW, RRF, VLM reranking, CQR/HyDE, and
grounded VQA are supporting mechanisms, not standalone novelty claims.

## Current paper structure

1. **Introduction** — VBS live-search constraints, system-level problem statement, and the three contributions: provenance contract, bounded live control, and fail-closed grounded VQA.
2. **VBS Setting and System-Level Delta** — VBS tasks and datasets, design goals, and the distinction between the legacy running snapshot and the instrumented source-index path.
3. **TGLTW-RMIT Architecture** — offline evidence/provenance, serving indexes, online dense/payload-text/RRF retrieval, and the web/DRES control plane.
4. **Task-Specific Live Pipelines** — KIS-V, KIS-T, KIS-C, AVS, and VQA routes that reuse a shared candidate and submission contract.
5. **Evaluation Protocol** — official live outcome, diagnostic offline replay, quality--latency ablations, provenance parity, grounding, and operational failures.
6. **Limitations and Conclusion** — migration boundary, live-judge dependence, model/API variability, reproducibility artifacts, and the final system-level claim.

The current draft is intentionally compact for the six-content-page plus up to
two-reference-page limit. If the benchmark is frozen before submission, replace
the protocol placeholders with measured result and failure tables without
changing the claim boundary.

## Minimum evaluation package before submission

### Retrieval

- Fixed query manifest and train/validation/test or held-out split.
- Dense-only, payload-text-only, RRF, RRF+HyDE, and optional VLM-rerank baselines.
- Recall@1/5/10/20, MRR, nDCG@10 and video/frame hit rate for KIS.
- AVS coverage, unique-video count, duplicate rate and operator time; keep live judging separate from automatic proxy metrics.

### VQA and grounding

- Candidate localization accuracy, answer exact/semantic accuracy, and grounded-answer rate.
- Missing-frame, malformed-response and timeout rates.
- Candidate/frame identity parity checks between the VLM input, API response and displayed media.

### Runtime

- Query processing, dense/payload-text search, fusion, reranking, VLM and end-to-end latency.
- p50/p95 for warm and cold paths, plus a bounded concurrent-load test.
- Peak memory/VRAM and index size for the frozen configuration.

### Ablations and artifacts

- One-factor ablations for HyDE, payload-text retrieval, RRF, temporal coherence, scene diversification, VLM reranking, grounded VQA and H-EAGLE-lite.
- Failure examples with source video/frame/timestamp and reason codes.
- Git SHA, model/provider, environment lock, Qdrant collection/schema version, query manifest hash, output hash and raw metric tables.

## Claims policy

Do not report a target configuration table as a result table. Distinguish
implementation paths from end-to-end validated task modes. Limit provenance
claims to the instrumented source-index path while the migration is open; legacy
records may provide source- or timestamp-level identity only. Do not describe
optional SigLIP, full enriched fields, or a re-indexed schema as part of the
measured live system until the corresponding collection and benchmark artifact
exists. Use `TO BE MEASURED` in draft result tables until then.

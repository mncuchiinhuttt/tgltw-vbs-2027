# AEGIS Academic Benchmark Implementation Plan

## Objective

Replace illustrative benchmark rows with reproducible, evidence-bounded offline replay measurements for AEGIS/VBS. The offline track is not an official VBS/DRES score.

## Research questions

1. How do dense retrieval, lexical search, secondary vision, RRF, temporal coherence, diversification, and VLM reranking change target ranking?
2. Does entity-preserving CQR plus clarification and negative feedback improve session-level retrieval and reduce ambiguity?
3. Does localization plus fail-closed validation reduce unsupported answers without conflating refusal with correctness?
4. What accuracy/latency Pareto frontier results from HNSW effort and concurrency?

## Implemented in this pass

- Model dispatch uses `WeMMEmbedding4BEmbedder` for local visual/text embeddings across inference, preprocessing, batch, and webapp entrypoints.
- The configured checkpoint is fixed to `tencent/WeMM-Embedding-4B`; Qwen is not an embedding substitute. The VLM remains a separate, optional component.
- `capture_benchmark_manifest.py` records code/query hashes, Git revision, Python version, and platform.
- Contract tests cover exact point matching, frame tolerance, denominator preservation, and official-score boundary.
- `academic_benchmark_report.md`, `academic_benchmark_audit.md`, and `benchmark_manifest.json` record current observations and limitations.

## Required before valid WeMM results

- Reindex the benchmark corpus using `tencent/WeMM-Embedding-4B`; dimensional equality alone does not prove model identity.
- Capture the exact checkpoint revision/hash and collection metadata.
- Rerun the direct replay, retrieval ablation, KIS-C, VQA, concurrency, and HNSW experiments after reindexing.
## Publication rules

Do not report fixed configuration rows as measured evidence. Do not call offline replay an official score. Keep `N/A` for unavailable metrics, preserve every denominator, and separate safety refusal from answer correctness. Use causal language only when paired per-query outputs show the component was actually toggled.

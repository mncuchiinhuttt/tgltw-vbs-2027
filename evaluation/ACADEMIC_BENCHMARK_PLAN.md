# AEGIS Academic Benchmark Implementation Plan

## Objective

Replace illustrative benchmark rows with reproducible, evidence-bounded offline replay measurements for AEGIS/VBS. The offline track is not an official VBS/DRES score.

## Research questions

1. How do dense retrieval, lexical search, secondary vision, RRF, temporal coherence, diversification, and VLM reranking change target ranking?
2. Does entity-preserving CQR plus clarification and negative feedback improve session-level retrieval and reduce ambiguity?
3. Does localization plus fail-closed validation reduce unsupported answers without conflating refusal with correctness?
4. What accuracy/latency Pareto frontier results from HNSW effort and concurrency?

## Implemented in this pass

- `run_comprehensive_ablation.py` measures only the production retrieval path and writes per-query ranks for M1--M5; M6 fails closed unless a VLM client is explicitly supplied.
- The runner reports explicit evaluable denominators and marks itself `MEASURED_RETRIEVAL_ONLY`.
- `capture_benchmark_manifest.py` records code/query hashes, Git revision, Python version, and platform.
- Contract tests cover exact point matching, frame tolerance, denominator preservation, and official-score boundary.
- `academic_benchmark_report.md`, `academic_benchmark_audit.md`, and `benchmark_manifest.json` record current observations and limitations.

## Remaining experiment work

- Add an explicit VLM client/configuration to M6 and retain per-candidate error states.
- Implement paired KIS-C sessions with fixed turn scripts; treat sessions, not turns, as independent units.
- Implement positive and negative VQA cases with frame localization, token-F1, grounded-answer, unsupported-answer, and safe-refusal metrics.
- Implement repeated warm latency trials for concurrency and HNSW `ef_search` values, including p50/p95/p99 and GPU telemetry.
- Expand independent annotated queries and freeze media/index checksums.
- Run paired bootstrap/randomization tests, Wilson intervals, and Holm correction after the larger set is available.

## Publication rules

Do not report fixed configuration rows as measured evidence. Do not call offline replay an official score. Keep `N/A` for unavailable metrics, preserve every denominator, and separate safety refusal from answer correctness. Use causal language only when paired per-query outputs show the component was actually toggled.

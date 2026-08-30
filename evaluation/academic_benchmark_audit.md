# Academic Benchmark Audit — AEGIS / VBS

## Scope

This benchmark is an **offline replay diagnostic**, not an official VBS/DRES leaderboard score. The current repository contains:

- `evaluation/run_rag_benchmark.py`: executable retrieval/VQA/KIS-C/telemetry runner.
- `evaluation/run_comprehensive_ablation.py`: runner shell for five ablation axes, but several values are currently hard-coded rather than measured.
- `evaluation/academic_benchmark_suite.py`: illustrative 5-axis table generator; its comments and values explicitly include simulation-style summaries.
- `queries/vbs_rag_benchmark.json`: 10 annotated examples: 3 KIS-T, 2 KIS-C, 3 VQA, 1 AVS, 1 KIS-V.
- `evaluation/vbs_rag_benchmark_results.json`: prior 10-query run with 9 retrieval-evaluable items, 2 ordinary VQA items, 1 fail-closed VQA item, 2 KIS-C scenarios.
- `evaluation/benchmark_real_output/comprehensive_ablation_summary.json`: prior output with 11 retrieval-evaluable items and measured retrieval values mostly unchanged from M1–M4; KIS-C/VQA/concurrency/HNSW values are emitted by fixed configuration rows in the runner.

## Current evidence and caveats

The prior 10-query replay reports: retrieval R@1 11.1%, R@5 22.2%, R@10 22.2%, MRR 0.1481 over 9 items; ordinary VQA EM 0% over 2 items; fail-closed safety 100% over the safety case; KIS-C R@1 0% over 2 scenarios; mean ambiguity reduction 0.024; p50/p95 total latency 28.266/38.711 seconds. These values are direct artifact evidence from the cached JSON, but they are underpowered and should not be generalized.

The prior comprehensive output reports M1–M4 each at R@1/R@5/R@10 = 9.1/9.1/9.1 and MRR 0.097, M5/M6 at 9.1/18.2/18.2 and MRR 0.141. This is useful as a diagnostic signal: the current ablation implementation does not actually activate distinct M1–M6 retrieval components. The KIS-C rows (C1–C5), VQA rows, concurrency rows, and HNSW rows are configuration literals in `run_comprehensive_ablation.py`; they must be labelled illustrative or replaced with instrumented measurements before publication.

## Proposed research questions

- **RQ1 — Retrieval:** How do dense retrieval, lexical fusion, secondary vision embeddings, reciprocal-rank fusion, temporal coherence/diversification, and VLM reranking affect target-hit ranking on a fixed, leakage-controlled VBS replay set?
- **RQ2 — Interaction:** Does entity-preserving CQR plus clarification/negative feedback improve target rank and reduce ambiguity relative to naive history concatenation, using paired sessions and fixed turn scripts?
- **RQ3 — Grounding:** Does evidence localization and fail-closed validation reduce unsupported answers while preserving answer accuracy on positive and negative VQA cases?
- **RQ4 — Operations:** What latency/throughput/recall trade-off is produced by concurrency and HNSW `ef_search` settings on the declared hardware and index snapshot?

## Hypotheses

- H1: Each retrieval component improves paired MRR or Recall@K over its immediate predecessor; report per-query deltas, bootstrap 95% CIs, and paired randomization tests rather than only aggregate percentages.
- H2: CQR and feedback reduce ambiguity and improve reciprocal rank relative to naive concatenation; test at the session level, not by treating turns as independent observations.
- H3: Fail-closed grounding lowers unsupported-answer rate; positive-answer EM and negative-case safe-refusal rate must be reported separately.
- H4: Increasing HNSW effort improves recall against exact search at a measurable latency cost; report Pareto points, not a single composite score.

## Metric contract

- Retrieval: Recall@1/5/10/20, MRR; target matching must use exact point/frame labels where available, with declared tolerance and a separate video-level metric.
- KIS-C: session-level target rank, MRR, Recall@K, ambiguity before/after, rank gain, turns-to-correct, and false-positive/rejected-item rate.
- VQA: frame localization hit, exact match, normalized token-F1, grounded-answer rate, unsupported-answer rate, safe-refusal rate on negative cases, and latency. Do not call `100%` safety a general claim when the denominator is one case.
- AVS: unique-video recall/coverage, duplicate rate, accepted-shot precision, and official judge outcome when available; ordinary Recall@K is insufficient for incomplete AVS labels.
- Operations: p50/p95/p99 latency, throughput, cold-start/init time, stage breakdown, GPU memory, and failure rate. Report repeated runs and exclude warm-up only by declared rule.

## Minimum publishable protocol

Freeze dataset manifest/checksums, query labels, index collection/configuration, model/provider revisions, hardware, random seeds, code commit, and raw per-query outputs. Use disjoint query/video splits where possible. Run at least 5 repeated latency trials after warm-up. Use paired bootstrap CIs and paired permutation/randomization tests for ranking comparisons; use Wilson intervals for binary VQA/safety rates. Correct multiple component comparisons with Holm or clearly mark exploratory tests. Keep all unavailable metrics as `N/A`; never replace them with simulated values.

## Publication status

The previous replay is **not valid evidence for the paper's WeMM-Embedding-4B claim** because the active environment selected Qwen3-VL-Embedding-2B. Configuration and dispatch are now WeMM-only, but the existing Qdrant points remain from the prior checkpoint. A fresh WeMM reindex and benchmark replay are mandatory before comparing performance or publishing new numbers.

## Current implementation status

The measured retrieval ablation runner writes per-query ranks and explicit evaluable denominators. KIS-C, VQA, concurrency, and HNSW remain separate follow-up experiments rather than fabricated rows.

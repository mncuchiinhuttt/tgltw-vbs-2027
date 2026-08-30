# AEGIS Academic Benchmark Report

## Material Passport

- **Artifact ID:** `aegis-vbs-offline-benchmark-2026-08-30`
- **Artifact type:** Offline replay benchmark and ablation audit
- **Status:** ANALYZED; not an official VBS/DRES score
- **Code revision:** current local `main` checkout at benchmark execution
- **Primary commands:**
  - `uv run python evaluation/run_rag_benchmark.py`
  - `uv run python evaluation/run_comprehensive_ablation.py --queries evaluation/eval_queries_real_v3c.json --dataset_dir datasets --output_dir evaluation/benchmark_real_output/current_run`
- **Runtime:** CUDA-enabled local environment; Qwen3-VL-Embedding-2B, OpenAI-compatible VLM `ag/gemini-3.5-flash-low`, YOLOE-26, Qdrant at `localhost:6333`
- **Query sources:** `queries/vbs_rag_benchmark.json` (10 annotated examples) and `evaluation/eval_queries_real_v3c.json` (18 annotated examples)

## Abstract

We evaluated AEGIS, an evidence-grounded multimodal retrieval system for Video Browser Showdown-style tasks, using offline replay over a small annotated query suite and a five-axis ablation harness. The direct 10-query replay completed successfully but produced weak retrieval evidence: five of ten cases entered the top 20, with KIS-T ranks 31, 4, and 10, no KIS-C target in the top 20, and no ordinary VQA answer exact matches. The fail-closed negative VQA case passed. Median end-to-end latency was 24.026 s and p95 latency was 34.146 s. The current ablation harness reproduced M5's improvement from R@5 9.1% to 18.2% on the 18-query manifest, but M1–M4 were identical and the KIS-C, VQA, concurrency, and HNSW rows are configuration literals rather than independent measured runs. Therefore, the evidence supports a diagnostic finding—not the stronger manuscript claims of 80–100% retrieval, 1.000 MRR, zero hallucination in a general VQA population, or 8.03x production speedup.

## Research questions and hypotheses

- **RQ1:** What is the effect of retrieval and reranking components on target ranking?
- **RQ2:** Does conversational context and negative feedback improve target rank and reduce ambiguity?
- **RQ3:** Does evidence localization plus fail-closed validation reduce unsupported answers?
- **RQ4:** What accuracy/latency trade-off results from HNSW effort and concurrency?

The preregistered-style hypotheses were directional component improvements (H1), paired conversational gains (H2), reduced unsupported answers without conflating refusal and correctness (H3), and a recall/latency Pareto frontier (H4). The present data are insufficient to test all four hypotheses with inferential confidence.

## Dataset and protocol

The direct replay contains 10 examples: 3 KIS-T, 2 KIS-C, 3 VQA (including one negative fail-closed case), 1 AVS, and 1 KIS-V. The extended manifest contains 18 examples: 10 KIS-T, 4 VQA, 2 KIS-C, 1 AVS, and 1 KIS-V. Labels identify target video, and in some cases timestamp/frame and answer. The system's current evaluator uses a target-video/frame tolerance for retrieval and a substring-style acceptable-answer check for VQA. These rules are local replay conventions, not official VBS rules.

No complete dataset checksum manifest, independent train/test partition, repeated latency trial set, or official DRES outcome log is present in these artifacts. Results are consequently reported with explicit denominators and scope limits.

## Direct offline replay results

| Task | n | Result | Interpretation |
|---|---:|---:|---|
| All replay cases entering target top-20 | 10 | 5/10 = 50.0% (Wilson 95% CI: 23.7–76.3%) | Broad diagnostic only; includes heterogeneous task semantics |
| KIS-T target rank | 3 | ranks 31, 4, 10; R@1 0/3, R@5 1/3, R@10 2/3 | Retrieval is not yet reliable at top-1 |
| KIS-C target rank | 2 | 0/2 in top-20 | No evidence of successful target retrieval in this run |
| Ordinary VQA exact answer | 2 | 0/2 (Wilson upper bound 65.8%) | Underpowered and failed on both positive cases |
| Negative VQA safe refusal | 1 | 1/1 (Wilson 95% CI: 20.7–100%) | Safety behavior observed in one negative case only |
| AVS annotated target | 1 | outside top-20 | No meaningful AVS estimate |
| KIS-V annotated target | 1 | outside top-20 | No meaningful KIS-V estimate |

Latency over all 10 cases: mean 18.006 s, median 24.026 s, minimum 0.202 s, maximum 34.146 s, p95 34.146 s. By task, mean latency was KIS-T 29.187 s, KIS-C 26.526 s, VQA 5.157 s, AVS 0.202 s, and KIS-V 23.771 s. These are one-run observations and are not stable service-level estimates.

## Ablation results

The current extended ablation command completed on 18 queries. For retrieval, 11 cases were considered evaluable. Measured output was:

| Configuration | R@1 | R@5 | R@10 | MRR | p50 latency |
|---|---:|---:|---:|---:|---:|
| M1 Dense only | 9.1% | 9.1% | 9.1% | 0.097 | 0.175 s |
| M2 Dense + BM25 | 9.1% | 9.1% | 9.1% | 0.097 | 0.164 s |
| M3 + SigLIP | 9.1% | 9.1% | 9.1% | 0.097 | 0.174 s |
| M4 + RRF | 9.1% | 9.1% | 9.1% | 0.097 | 0.163 s |
| M5 + temporal/diversification | 9.1% | 18.2% | 18.2% | 0.141 | 0.168 s |
The retrieval ablation runner has now been corrected to fail closed when M6 is configured without a VLM client. Its smoke run therefore reports M6 as an explicit unavailable configuration rather than silently claiming a zero-score rerank; this is an intentional integrity guard.

The M5–M1 R@5 difference is one additional hit among 11 evaluable queries (absolute +9.1 percentage points). No paired per-query outputs are emitted for each configuration, so a paired significance test and confidence interval for the component effect cannot be calculated from this artifact. More importantly, the runner's flags do not activate distinct BM25, SigLIP, RRF, or VLM branches; M1–M4 and M5–M6 therefore cannot be interpreted as valid causal component ablations.

KIS-C rows report C1–C5 as 0/30/60/90/100% R@1 and ambiguity 0.82/0.74/0.58/0.42/0.24, but the runner defines these values as a fixed table. VQA rows report 55/80/100% EM and 38/14/0% hallucination, concurrency rows report 1.00x/3.73x/8.03x, and HNSW rows report 97.8/99.2/99.9/100% recall-versus-exact; these are also fixed configuration values in the runner. They must be marked illustrative until replaced with instrumented repeated measurements.

## Statistical interpretation and fallacy scan

1. **Denominator inflation:** avoid presenting 100% from one negative VQA case as population safety.
2. **Metric conflation:** safe refusal is not answer correctness; AVS coverage is not ordinary Recall@K.
3. **Pseudo-replication:** KIS-C turns are nested in sessions and cannot be treated as independent queries.
4. **Unpaired ablation bias:** aggregate deltas without per-query paired outcomes cannot establish component causality.
5. **Leakage risk:** query examples and target media/index provenance need a frozen split and checksum manifest.
6. **Multiple comparisons:** five ablation families require a declared correction plan.
7. **Latency censoring:** one run per query cannot support p95 service claims; warm-up and provider variance are unreported.
8. **Measurement validity:** current frame/video tolerance and substring answer match are local conventions.
9. **Selective reporting:** failed/out-of-top-20 cases must remain in denominators.
10. **Unsupported causal language:** current results support observations, not claims that a named component caused gains.
11. **Composite-score opacity:** the cached `overall_rag_score` is a hand-weighted aggregate and should not replace task-specific metrics.

## Reproducibility and next experiment

Before publication, freeze: dataset and media checksums, Qdrant collection snapshot/configuration, model/provider revisions, hardware, environment lock, query-label version, random seeds, and raw per-query ranked lists. Expand the annotated replay set with balanced, independently adjudicated examples per task. For each ablation, emit per-query outputs from genuinely toggled configurations on the same candidates. Run at least five warm trials per query/configuration for latency and report median/p95 with bootstrap intervals. Use paired bootstrap or randomization tests for ranking metrics, Wilson intervals for binary rates, and Holm correction for component comparisons. Keep unavailable values as `N/A`.

## Conclusion

The present benchmark is valuable as a failure-oriented audit: it exposes retrieval and grounding weaknesses, while confirming one narrow fail-closed behavior. It is not sufficient evidence for the current paper's optimistic headline numbers. The scientifically defensible next step is instrumenting the ablation runner and enlarging the labeled replay set, not copying the fixed rows into publication tables.

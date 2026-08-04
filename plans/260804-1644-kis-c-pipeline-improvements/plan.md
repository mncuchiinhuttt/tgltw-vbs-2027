---
title: "KIS-C Pipeline Improvements (zero extra LLM calls)"
description: "Five KIS-C upgrades - few-shot CQR, facet-driven clarification, clarification-answer boost, score-margin ambiguity, Rocchio-into-CQR - all at today's LLM-call cost."
status: implemented + code-reviewed (no critical/high findings); manual live-VLM verification checklist still pending a real backend/VLM session
priority: P2
effort: 5h
branch: main
tags: [kis-c, conversational-search, retrieval, prompt-engineering, vbs-2027]
created: 2026-08-04
---

# KIS-C Pipeline Improvements

Five scoped upgrades to the conversational (KIS-C) path. **Hard constraint: the
per-turn LLM/VLM call count must stay exactly as today** (CQR 1 + HyDE 1 +
clarification 0-or-1 + rerank). Every item is prompt engineering or pure Python.
Consequence: DMQR-RAG multi-query rewrite, multi-hypothesis/beam rewriting,
ConvGQR rewrite+expansion, and information-gain question selection are all
**out of scope** (already rejected - each adds a call).

## Phases

| # | Phase | Scope (items) | Effort | Status |
|---|-------|---------------|--------|--------|
| 01 | [Pure scoring logic](phase-01-pure-kis-c-scoring-logic.md) | #4 score-margin ambiguity, #3 clarification-answer boost (function only) | 1h | done |
| 02 | [Conversational prompt module](phase-02-conversational-prompt-module.md) | #1 few-shot CQR, #2 facet-driven clarification, #5 feedback-in-prompt | 1.5h | done |
| 03 | [Backend + frontend wiring](phase-03-backend-session-and-wiring.md) | #3 wiring, #5 feedback -> history | 1.5h | done |
| 04 | [Tests + docs](phase-04-tests-and-verification.md) | unit tests, manual checklist, CHANGELOG | 1h | done |

## Dependencies

- 01 and 02 are independent (different new files) - can run in parallel.
- 03 depends on **both** 01 and 02 (imports both new modules).
- 04 depends on 03.

## New files (both pure, dependency-free, unit-testable)

- `inference-code/search/kis_c_scoring.py` - ambiguity signals + clarification-answer boost.
- `inference-code/search/conversational_context.py` - CQR/clarification prompt builders + history formatting + feedback recording.
- `tests/test_kis_c_scoring.py`, `tests/test_conversational_context.py`.

## Modified files

- `inference-code/search/hybrid_search.py` (`compute_ambiguity_score` delegates to the new pure module; same float return, same signature).
- `inference-code/search/query_processor.py` (`rewrite_query_cqr`, `generate_clarification_question` delegate prompt building).
- `webapp/backend/main.py` (session state fields, `/api/search` boost wiring, `/api/feedback` history annotation, `SearchRequest.clarification_answer`).
- `webapp/frontend/src/App.tsx` (one line: send `clarification_answer` alongside the composed query).
- `CHANGELOG.md`.

## Key architectural decisions

1. **Pure logic lives in new module-level functions, not new `HybridSearcher`
   methods** - matches the `_check_avs_duplicate` / `filter_asr_segments`
   precedent ("factored out so it's testable without importing fastapi/qdrant").
   `HybridSearcher.__init__` builds a `QdrantClient`, so a method is not
   dependency-free-testable; `compute_ambiguity_score`'s math moves out for the
   same reason while its public method signature stays unchanged.
2. **Prompt strings live in a dedicated module** - makes the three prompt-only
   items (#1, #2, #5) structurally testable (placeholders, example count, facet
   instruction, feedback lines) instead of "untestable without an LLM", and
   keeps `query_processor.py` under the 200-line convention.
3. **The clarification answer arrives as an explicit request field**, not by
   string-parsing the frontend's `"Additional detail from operator: ..."`
   suffix. The composed query keeps going to retrieval unchanged.
4. **No fast path / no branch that skips search.** The boost is additive on top
   of the normal per-turn RRF-fused pool (KISS).

## Research inputs

`plans/260804-0415-kis-c-conversational-retrieval-research/research/researcher-{a,b,c}-*.md`

## Next Steps (after implementation)

After implement -> test -> code-review, the **orchestrating session (not the
planner)** updates the team's Notion "Our method (VBS)" page's KIS-C section to
document these five changes. Also see phase 04 for the CHANGELOG entry.

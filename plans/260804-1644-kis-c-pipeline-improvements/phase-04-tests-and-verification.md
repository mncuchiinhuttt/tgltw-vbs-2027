# Phase 04 - Tests, Manual Verification, Docs

## Context Links

- Test template: `tests/test_asr_segment_filter.py` (pure logic, stub dicts, no model/network, `sys.path` append + import of the real production module, runnable under pytest **or** as a plain script)
- Phase 01 (`phase-01-pure-kis-c-scoring-logic.md`), Phase 02 (`phase-02-conversational-prompt-module.md`), Phase 03 (`phase-03-backend-session-and-wiring.md`)
- `pyproject.toml` - `pytest` lives in the `dev` dependency group
- `CHANGELOG.md` - current top entry `[1.16.0] - 2026-08-04`
- Plan overview: `plan.md`

## Overview

- **Priority:** P2
- **Status:** done
- **Effort:** ~1h
- Unit tests for every new pure function, an explicit manual checklist for the
  parts that genuinely cannot be tested without a live LLM, and the CHANGELOG
  entry.

## Key Insights

- Both new modules are stdlib-only by design, so the new tests need **no**
  dependency group at all - `python3 tests/test_kis_c_scoring.py` works on a bare
  interpreter, same as the existing test file's script mode.
- `inference-code` contains a hyphen, so it is not importable as a package. Tests
  must put `<repo>/inference-code` on `sys.path` and then
  `from search.kis_c_scoring import ...` - exactly how `hybrid_search.py` and
  `webapp/backend/main.py` already reach these modules.
- Items #1, #2 and the prompt half of #5 are prompt changes. What is testable:
  **prompt structure** (required placeholders present, correct example count,
  facet instruction present, feedback lines rendered, no crash on empty input).
  What is not: whether the LLM actually produces better rewrites/questions. That
  gap is closed by a manual checklist, **documented as a known limitation** - do
  not stub a fake VLM and call it coverage.
- Do not test `webapp/backend/main.py` wiring with a fake VLM/Qdrant. The wiring
  is ~30 lines of orchestration; its verification is the manual round-trip in
  phase 03 step 9 plus `code-reviewer`.

## Requirements

### Functional

1. `tests/test_kis_c_scoring.py` covers the ambiguity signals and the
   clarification-answer boost.
2. `tests/test_conversational_context.py` covers the prompt builders, history
   formatting, and feedback recording.
3. Both files follow the existing template: module docstring stating what is
   stubbed and how to run, `sys.path` setup, plain `assert`s, `_run_all()`
   script-mode fallback with a non-zero exit on failure.
4. Every test uses stub dicts only. No network, no model, no Qdrant, no LLM.
5. All tests pass; existing `tests/test_asr_segment_filter.py` still passes.

### Non-functional

- Deterministic (no randomness, no time/locale dependence).
- Fast (<1s total).
- No new dependency added to `pyproject.toml`.

## Architecture

```
tests/
  test_asr_segment_filter.py        (existing, untouched)
  test_kis_c_scoring.py             (new)  -> inference-code/search/kis_c_scoring.py
  test_conversational_context.py    (new)  -> inference-code/search/conversational_context.py
```

Runner (both work):
- `uv run --group dev python3 -m pytest tests/ -q`
- `python3 tests/test_kis_c_scoring.py` (script mode, zero deps)

Note: `pytest` is in the `dev` group, not `preprocessing`. If a combined run is
wanted alongside the ASR test's imports, use
`uv run --group dev --group preprocessing python3 -m pytest tests/ -q` and
confirm which invocation actually resolves in this environment before writing it
into the CHANGELOG.

## Related Code Files

**Create**

- `tests/test_kis_c_scoring.py`
- `tests/test_conversational_context.py`

**Modify**

- `CHANGELOG.md`

## Implementation Steps

1. `tests/test_kis_c_scoring.py` - helper factory:
   ```python
   def _cand(cid="p1", video="V001.mp4", score=0.05, text_blob="a red car on a street"):
       return {"id": cid, "rrf_score": score, "payload": {"source_file": video, "text_blob": text_blob}}
   ```

2. Ambiguity tests:
   - `test_distinct_video_ratio_all_same_video` -> 0.0-ish (ratio = 1/len).
   - `test_distinct_video_ratio_all_different` -> 1.0.
   - `test_distinct_video_ratio_empty_returns_zero`.
   - `test_score_margin_ambiguity_runaway_winner_is_low` (scores 1.0, 0.1 -> ~0.1).
   - `test_score_margin_ambiguity_tied_scores_is_high` (1.0, 1.0 -> 1.0).
   - `test_score_margin_ambiguity_single_candidate_returns_zero`.
   - `test_score_margin_ambiguity_zero_scores_returns_zero`.
   - `test_score_margin_ambiguity_ignores_input_order` (unsorted input -> same value).
   - `test_combine_ambiguity_signals_is_within_unit_range` (incl. out-of-range inputs clamped).
   - `test_combined_score_tied_distinct_pool_still_triggers` - 10 distinct videos,
     equal scores -> `>= 0.7` (the live `AMBIGUITY_THRESHOLD`), i.e. the
     behaviour operators depend on is preserved.
   - `test_combined_score_clear_winner_does_not_trigger` - 10 distinct videos, one
     runaway score -> `< 0.7`.
   - `test_combined_score_single_candidate_does_not_trigger` - documents the
     intentional fix of the old `1 distinct / 1 = 1.0` false trigger.

3. Boost tests:
   - `test_boost_raises_matching_candidate_above_higher_scored_one` - candidate B
     (lower `rrf_score`, caption mentions "red") ends up ranked above A after
     `answer_text="the car was red"`.
   - `test_boost_ignores_candidates_not_in_prior_ids` - a matching candidate whose
     id is absent from `prior_candidate_ids` keeps its exact score.
   - `test_boost_empty_answer_is_noop` - list identity/order and every score
     unchanged.
   - `test_boost_empty_prior_ids_is_noop`.
   - `test_boost_no_token_overlap_leaves_scores_unchanged`.
   - `test_boost_never_lowers_a_score` - every score `>=` its pre-call value.
   - `test_boost_vietnamese_answer_matches_vietnamese_caption` - e.g. answer
     `"áo màu đỏ"` vs `text_blob` `"người mặc áo đỏ đi trên phố"` -> boosted
     (guards the `MIN_TOKEN_LEN=2` + `re.UNICODE` decisions).
   - `test_boost_falls_back_to_caption_when_text_blob_missing` - payload with
     `caption`/`ocr_text`/`detected_objects` only.
   - `test_boost_handles_missing_payload_and_missing_scores` - no `KeyError`.
   - `test_boost_stopwords_alone_do_not_boost` - answer `"it is in the"` -> no-op.

4. `tests/test_conversational_context.py`:
   - `test_build_cqr_prompt_contains_latest_query_and_history`.
   - `test_build_cqr_prompt_has_at_least_three_examples` - count
     `"Rewritten Query:"` occurrences `>= 4` (3 examples + live slot).
   - `test_build_cqr_prompt_renders_rejected_and_accepted_lines` - history turn
     with `rejected`/`accepted` -> descriptions appear in the prompt.
   - `test_build_cqr_prompt_omits_empty_system_answer_line` - a turn without
     `answer` produces no bare `System:` line.
   - `test_build_cqr_prompt_empty_history_does_not_raise`.
   - `test_format_history_handles_none`.
   - `test_build_clarification_prompt_lists_all_summaries`.
   - `test_build_clarification_prompt_instructs_facet_identification` - asserts the
     differing-attribute instruction and the "only the question" instruction are
     present (case-insensitive substring checks on the agreed wording; keep the
     asserted substrings short so harmless prompt rewording doesn't break CI).
   - `test_build_clarification_prompt_empty_summaries_does_not_raise`.
   - `test_describe_candidates_prefers_caption_then_video_then_id`.
   - `test_describe_candidates_truncates_and_caps_at_limit`.
   - `test_describe_candidates_deduplicates`.
   - `test_record_feedback_in_history_no_history_is_noop`.
   - `test_record_feedback_in_history_writes_accepted_and_rejected`.
   - `test_record_feedback_in_history_accumulates_without_duplicates` - call twice,
     assert no duplicates and length `<= limit`.
   - `test_record_feedback_in_history_unknown_id_still_recorded` - falls back to
     the raw id rather than dropping the signal.

5. Run: `uv run --group dev python3 -m pytest tests/ -q` **and** both files in
   script mode. Fix real failures - never weaken an assertion to go green.

6. Manual verification checklist (record the outcome in the phase-03 PR/report,
   not in a new file). **NOT YET EXECUTED** - no live backend/VLM was available
   in the implementing session; requires a human operator running the actual
   webapp against a real VLM. Do not check these off without actually running
   them:
   - [ ] Vague EN query -> clarification question is about an attribute that
         actually differs across the shown results, not a generic one.
   - [ ] Vague VI query -> question comes back **in Vietnamese**.
   - [ ] Question output contains no leaked facet reasoning/preamble.
   - [ ] Multi-turn EN: turn 2 uses a pronoun -> logged `CQR Rewrite:` line shows
         it resolved against turn 1.
   - [ ] Multi-turn VI: implicit reference resolved; rewrite stays Vietnamese.
   - [ ] Mark 2 results irrelevant via `/api/feedback`, then search again -> the
         next rewrite does not re-describe the rejected reading.
   - [ ] Answer a clarification -> response has `clarification_boost_applied:
         true`, the answered-about candidate moves up, and no second (redundant)
         question is asked.
   - [ ] Per-turn latency is not visibly worse than before the few-shot prompt.

7. `CHANGELOG.md` - add a new version section above `[1.16.0]`, matching the
   existing style (bold lead-in, mechanism, file paths, verified citations).
   Cover: few-shot CQR (arXiv:2502.15009); facet-driven clarification (Sekulic et
   al., zero-shot variant); clarification-answer boost (Sekulic et al.
   arXiv:2008.03717, +18% recall/+12% nDCG@3 **in that paper's setting -
   indicative only**); score-margin ambiguity augmentation (+ the intentional
   single-candidate false-trigger fix); Rocchio feedback threaded into CQR context
   (Exquisitor VBS 2024/2025, lightest prompt-only form). State explicitly that
   **all five keep the per-turn LLM/VLM call count unchanged**, and that
   DMQR-RAG-style multi-query rewriting, multi-hypothesis rewriting, ConvGQR
   expansion and information-gain question selection remain rejected for adding a
   call. List the two new test files.
   Note: this repo has **no `docs/` directory** (the documentation-management rule
   assumes one) - `CHANGELOG.md` at the repo root is the only doc to update. Do
   not create a `docs/` tree as part of this plan.

## Todo List

- [x] `tests/test_kis_c_scoring.py` - ambiguity signal tests
- [x] `tests/test_kis_c_scoring.py` - boost tests (incl. Vietnamese + no-op cases)
- [x] `tests/test_conversational_context.py` - prompt builder tests
- [x] `tests/test_conversational_context.py` - history + feedback recording tests
- [x] Both files runnable under pytest and as plain scripts
- [x] Full suite green (new + existing ASR test)
- [ ] Manual verification checklist executed and results recorded (blocked - needs a live backend/VLM, not available in the implementing session)
- [x] `CHANGELOG.md` entry

## Success Criteria

- `uv run --group dev python3 -m pytest tests/ -q` -> all pass, 0 skipped, 0 xfail.
- `python3 tests/test_kis_c_scoring.py` and `python3 tests/test_conversational_context.py`
  pass on a bare interpreter (proves the new modules are dependency-free).
- Every public function added in phases 01 and 02 has at least one test, including
  one empty/malformed-input test each.
- The two threshold-behaviour tests (tied pool still triggers at 0.7 / clear
  winner does not) are present - they are the regression guard for the ambiguity
  change.
- No mocks, no fake VLM, no fake Qdrant anywhere in `tests/`.
- Manual checklist fully executed; any failing item fixed and re-verified, not
  waived.

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Substring assertions on prompt wording make CI brittle | Assert only short, load-bearing fragments (e.g. `"differs"`, `"ONLY the question"`) plus counts - not whole sentences |
| Hard-coded expected numbers drift if `MARGIN_WEIGHT` is retuned | Assert against the threshold (`>= 0.7` / `< 0.7`) and inequalities/ordering rather than exact floats, except where a value is definitionally exact (0.0 / 1.0) |
| Temptation to stub the VLM to "test" the prompts | Explicitly forbidden here; the limitation is documented and covered by the manual checklist |
| `pytest` group mismatch (`dev` vs `preprocessing`) breaks the documented command | Verify the exact invocation before writing it into the CHANGELOG; script mode is the always-works fallback |
| Manual checklist quietly skipped under time pressure | It is a Todo item and a Success Criterion; results are recorded in the phase-03 report |

## Security Considerations

- Tests use synthetic stub payloads only - no real dataset frames, no captions
  from private data, no credentials, no `.env` reads, no network. Nothing new is
  committed that could leak competition data.

## Next Steps

- Run `code-reviewer` over the full diff (phases 01-03 + tests).
- **The orchestrating session (not the planner) then updates the team's Notion
  "Our method (VBS)" page, KIS-C section**, to document these five changes -
  including that all five respect the zero-extra-call rule, and keeping
  DMQR-RAG / multi-hypothesis / ConvGQR / information-gain selection listed under
  "Không áp dụng (research xong, quyết định bỏ)".
- Optional follow-up (out of scope): tune `AMBIGUITY_THRESHOLD` / `MARGIN_WEIGHT`
  on recorded VBS-2025 KIS-C tasks; split the 1176-line `webapp/backend/main.py`.

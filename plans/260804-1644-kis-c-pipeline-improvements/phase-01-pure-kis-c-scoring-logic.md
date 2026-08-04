# Phase 01 - Pure KIS-C Scoring Logic

## Context Links

- Research: `../260804-0415-kis-c-conversational-retrieval-research/research/researcher-b-conversational-rewriting-sota.md` (§5 "Score Entropy & Margin")
- Research: `../260804-0415-kis-c-conversational-retrieval-research/research/researcher-c-clarifying-question-sota.md` (§4 "Clarification Answer -> Direct Re-rank", Sekulic et al. arXiv:2008.03717)
- Code: `inference-code/search/hybrid_search.py` (`compute_ambiguity_score` L295, `temporal_coherence_boost` L251, `diversify_by_scene` L317)
- Payload shape: `preprocessing/main.py` L346-366 (`caption`, `scene_narrative`, `ocr_text`, `detected_objects`, `actions`, `text_blob`)
- Plan overview: `plan.md`

## Overview

- **Priority:** P2 (blocks phase 03)
- **Status:** done
- **Effort:** ~1h
- Scope items **#4** (score-margin augmentation of the ambiguity score) and the
  **function half of #3** (boost candidates using the clarification answer).
  Both are pure Python over data already in memory - zero LLM/VLM calls, zero
  Qdrant calls.

## Key Insights

- `compute_ambiguity_score` today is `distinct_videos / top_n` only. It is
  *count*-based and completely ignores how the scores are distributed, so
  "10 videos, one runaway winner" and "10 videos, all tied" both score 1.0.
- Margin is simpler and more interpretable than entropy (researcher-b) and needs
  only two numbers. Chosen over entropy for that reason (KISS).
- The margin signal is meaningful **at the call site** because
  `compute_ambiguity_score` runs *after* `temporal_coherence_boost`, which
  deliberately spreads the score distribution (a genuine event cluster gets a
  large additive boost). On raw RRF scores alone the top-1/top-2 gap would be
  near-zero and the signal would be useless.
- Side benefit: today a **single**-candidate pool scores 1.0 (1 distinct / 1) and
  wrongly triggers a clarification question. With the margin term undefined ->
  treated as 0.0 ambiguity, the combined score falls below the 0.7 threshold.
  Intentional fix; call it out in the CHANGELOG.
- `HybridSearcher.__init__` constructs a `QdrantClient`, so anything defined as a
  method on it cannot be unit-tested dependency-free. Hence a new pure module.
  Precedent: `_check_avs_duplicate` in `webapp/backend/main.py` and
  `preprocessing/audio/asr_segment_filter.py`.
- The boost must be **additive only** (never demote) and scaled relative to the
  pool's own top score, exactly like `temporal_coherence_boost` scales by
  neighbour score sums - so it composes with RRF scores of any magnitude.

## Requirements

### Functional

1. `compute_ambiguity_score(candidates, top_n=10)` keeps its signature and keeps
   returning a single `float` in `[0.0, 1.0]`. **No call-site change** in
   `webapp/backend/main.py` (`AMBIGUITY_THRESHOLD` comparison stays as-is).
2. The returned float combines two signals: the existing distinct-video ratio and
   a new normalized top-1/top-2 score margin.
3. New `boost_by_clarification_answer(candidates, prior_candidate_ids, answer_text, ...)`
   returns a score-sorted list; boosts only candidates whose id is in
   `prior_candidate_ids` and whose payload text overlaps the answer tokens.
4. Both functions are total: empty/None/malformed inputs return sane values, never
   raise (KIS-C runs on a live 7-minute competition clock).

### Non-functional

- New module imports **stdlib only** (`re`). No numpy, no qdrant_client, no config.
- Both functions O(n) / O(n log n) over <= a few hundred candidates.
- New module under 200 lines.

## Architecture

```
webapp/backend/main.py  /api/search  (type 1)
  merge_rrf -> temporal_coherence_boost -> diversify_by_scene
     |
     +-- [phase 03] boost_by_clarification_answer(...)   <-- kis_c_scoring (pure)
     |
     +-- searcher.compute_ambiguity_score(candidates)    <-- HybridSearcher method
     |        delegates to -> distinct_video_ratio() + score_margin_ambiguity()
     |                        + combine_ambiguity_signals()   (kis_c_scoring, pure)
     |
     +-- reranker.rerank_type1(...)
```

`hybrid_search.py` keeps the public method; only the arithmetic moves.

## Related Code Files

**Create**

- `inference-code/search/kis_c_scoring.py`

**Modify**

- `inference-code/search/hybrid_search.py` - `compute_ambiguity_score` body + docstring.

**Delete** - none.

## Implementation Steps

1. Create `inference-code/search/kis_c_scoring.py` with a module docstring stating:
   pure logic, stdlib only, no LLM/VLM/Qdrant calls, imported by
   `hybrid_search.py` and `webapp/backend/main.py`.

2. Module constants:
   - `MARGIN_WEIGHT = 0.5` - weight of the margin signal in the combined score.
   - `CLARIFICATION_BOOST_WEIGHT = 0.5` - max fraction of the pool's top score a
     fully-matching candidate can gain.
   - `MIN_TOKEN_LEN = 2` - Vietnamese content words are often 2 chars ("áo",
     "xe"), so do NOT use 3.
   - `STOPWORDS: frozenset` - function words only, EN + VI. EN: the, a, an, and,
     or, of, in, on, at, to, is, are, was, were, it, its, this, that, with, for,
     có. VI: là, và, của, được, thì, mà, các, những, một, cái, đang, ở, với, cho,
     rồi, thấy. **Do not** add content words like `người`/`xe`/`màu`.

3. `def distinct_video_ratio(candidates: list, top_n: int = 10) -> float` -
   verbatim current logic (`{payload["source_file"]}` over the top slice,
   `0.0` when the slice is empty). Use `.get("payload") or {}` so a malformed
   candidate can't raise.

4. `def score_margin_ambiguity(candidates: list, top_n: int = 10, score_key: str = "rrf_score") -> float`:
   - take the top slice, read scores via `c.get(score_key, 0.0)`, sort desc
     (do not assume the caller pre-sorted).
   - fewer than 2 scores, or `top1 <= 0` -> return `0.0` (no evidence of
     ambiguity; this is what fixes the single-candidate case).
   - `relative_margin = (top1 - top2) / top1`, clamp to `[0.0, 1.0]`.
   - return `1.0 - relative_margin` (big lead -> low ambiguity).

5. `def combine_ambiguity_signals(distinct_ratio: float, margin_ambiguity: float, margin_weight: float = MARGIN_WEIGHT) -> float`:
   - `(1 - margin_weight) * distinct_ratio + margin_weight * margin_ambiguity`,
     clamped to `[0.0, 1.0]`. Weighted average (not OR-of-thresholds) so the
     result stays a single interpretable float on the same 0-1 scale the existing
     `AMBIGUITY_THRESHOLD=0.7` env knob is already tuned against.

6. `def candidate_match_text(payload: dict) -> str`:
   - prefer `payload["text_blob"]` (already concatenates caption + scene
     narrative + OCR + object labels + actions + speech at index time - DRY).
   - fall back to joining `caption`, `scene_narrative`, `ocr_text`, and
     `label` values from `detected_objects` (list of dicts) plus `actions`, for
     candidates indexed before `text_blob` existed.
   - return lowercased.

7. `def tokenize_answer(text: str) -> set` - `re.findall(r"\w+", text.lower(), re.UNICODE)`
   (`\w` + `re.UNICODE` already covers Vietnamese diacritics), drop tokens shorter
   than `MIN_TOKEN_LEN` and tokens in `STOPWORDS`, return a `set`.

8. `def boost_by_clarification_answer(candidates, prior_candidate_ids, answer_text, boost_weight=CLARIFICATION_BOOST_WEIGHT, score_key="rrf_score") -> list`:
   - Docstring must cite Sekulic et al. arXiv:2008.03717 ("clarification answer
     -> direct re-rank"; +18% recall / +12% nDCG@3 **in that paper's setting -
     indicative, not a guarantee here**) and state explicitly that this is an
     additive boost on the current turn's normal RRF pool, *not* a fast path that
     skips search.
   - Guard: no `candidates`, no `answer_text` tokens -> return `candidates`
     unchanged (same object order, no re-sort, so it is a true no-op).
   - `prior = set(prior_candidate_ids or ())`. If `prior` is empty -> return
     `candidates` unchanged (nothing was retained from the clarification turn).
   - `top_score = max(scores) or 1.0` (guard <= 0 -> `1.0`).
   - For each candidate with `c["id"] in prior`: `overlap = len(tokens & candidate_tokens) / len(tokens)`
     where `candidate_tokens` comes from `tokenize_answer(candidate_match_text(payload))`
     (reuse the same tokenizer on both sides so matching is symmetric - avoids
     substring false positives like "red" matching "reduce").
     Then `c[score_key] = c.get(score_key, 0.0) + boost_weight * overlap * top_score`.
     Also set `c["clarification_overlap"] = round(overlap, 3)` when `overlap > 0`
     so the operator/logs can see why a result moved (mirrors `matched_via`
     explainability).
   - Mutates candidate dicts in place (same as `temporal_coherence_boost`) and
     returns the list sorted by `score_key` desc.

9. Edit `hybrid_search.py`:
   - No new top-level import gymnastics needed - `sys.path` already includes
     `inference-code`, so `from search.kis_c_scoring import (...)` works
     alongside the existing `from config import ...`.
   - Rewrite `compute_ambiguity_score`'s body as three delegated calls +
     `combine_ambiguity_signals`. Keep the existing docstring's CAR /
     TrustNLP framing, append: what the margin term adds, why margin over
     entropy, why it's still one float, and the single-candidate fix.
   - Keep one `print(...)` of both component signals for live-session
     debuggability (the file already prints in `__init__`).

10. Compile check: `uv run --group inference python3 -c "import sys; sys.path.insert(0,'inference-code'); import search.kis_c_scoring"`
    and `python3 -m py_compile inference-code/search/hybrid_search.py`.

## Todo List

- [x] Create `inference-code/search/kis_c_scoring.py` with module docstring + constants
- [x] `distinct_video_ratio`
- [x] `score_margin_ambiguity`
- [x] `combine_ambiguity_signals`
- [x] `candidate_match_text` + `tokenize_answer`
- [x] `boost_by_clarification_answer`
- [x] Rewire `HybridSearcher.compute_ambiguity_score` (signature + float return unchanged)
- [x] `py_compile` both files clean

## Success Criteria

- `compute_ambiguity_score(candidates)` still returns a `float`; `webapp/backend/main.py:398` untouched by this phase.
- All-tied 10-distinct-video pool -> score close to 1.0 (still triggers at 0.7).
- One runaway winner among 10 distinct videos -> score at/below ~0.5 (no longer triggers).
- Single-candidate pool -> score at/below 0.5 (no longer triggers).
- `boost_by_clarification_answer` with empty answer or empty `prior_candidate_ids` is a byte-for-byte no-op on the input list.
- `import search.kis_c_scoring` succeeds with only stdlib available.
- Zero new LLM/VLM/Qdrant calls (grep the new module for `vlm`, `client`, `embed` -> no hits).

## Risk Assessment

| Risk | Mitigation |
|---|---|
| `MARGIN_WEIGHT=0.5` halves the effective scale and silently stops all clarification questions | Success criteria above pin the expected values at both extremes; phase 04 unit-tests them; weight is a kwarg + module constant so it can be retuned without touching call sites |
| `(top1-top2)/top1` normalization is scale-dependent and RRF gaps are tiny pre-boost | Documented: the only production call site runs after `temporal_coherence_boost`. If the boost is ever removed, revisit |
| Keyword overlap misses paraphrases ("crimson" vs "red") | Accepted limitation, documented in the docstring - the point is a cheap additive nudge, the reranker still decides the final order |
| Vietnamese stopword list drops content words | Restricted to function words; explicit "do not add" list in step 2 |
| Mutating candidate dicts in place surprises a caller | Same convention as the existing `temporal_coherence_boost`; documented in the docstring |

## Security Considerations

- No auth surface. `answer_text` is operator-supplied free text used only for
  tokenization and dict lookups - never in a query, filter, `eval`, or shell
  string, so no injection path.
- `re.findall(r"\w+")` on operator text is linear, no catastrophic-backtracking risk.
- No secrets, no logging of payload contents beyond the existing debug prints.

## Next Steps

- Phase 03 imports `boost_by_clarification_answer` in `/api/search`.
- Phase 04 adds `tests/test_kis_c_scoring.py`.

# Phase 03 - Backend Session State + Pipeline Wiring

## Context Links

- Phase 01 (`phase-01-pure-kis-c-scoring-logic.md`) - provides `boost_by_clarification_answer`
- Phase 02 (`phase-02-conversational-prompt-module.md`) - provides `record_feedback_in_history`
- Code: `webapp/backend/main.py` L84-104 (session state + `AMBIGUITY_THRESHOLD`), L312-521 (`/api/search`), L523-561 (`/api/feedback`), L187-208 (`SearchRequest`/`FeedbackRequest`)
- Code: `webapp/frontend/src/App.tsx` L232-253 (composes `requestQuery` with the clarification answer), L294 (stores the returned question)
- Research: researcher-c §4 (Sekulic et al. arXiv:2008.03717), researcher-a technique #1 (Exquisitor)
- Plan overview: `plan.md`

## Overview

- **Priority:** P2
- **Status:** done
- **Effort:** ~1.5h
- Wiring half of scope items **#3** (clarification-answer boost) and **#5**
  (Rocchio feedback -> CQR context). Orchestration only: no new LLM/VLM call, no
  new Qdrant call, no new branch that skips search.

## Key Insights

- The clarification round-trip already exists end-to-end in the UI: backend
  returns `clarification`, the frontend shows the amber banner, the operator types
  an answer, and the **next** `/api/search` gets
  `"<query>\nAdditional detail from operator: <answer>"` as `request.query`
  (`App.tsx` L234-236). The answer text is therefore *already* reaching CQR and
  retrieval - what is missing is (a) knowing which candidates the question was
  about and (b) an isolated handle on the answer text.
- Do **not** recover the answer by string-splitting on
  `"Additional detail from operator: "`. Add an explicit optional
  `clarification_answer` field; the composed query keeps flowing to retrieval
  exactly as today so retrieval behaviour is unchanged by this phase.
- `/api/feedback` receives point **ids**, not payloads, and `get_point_vector`
  returns vectors only. Rather than adding a Qdrant payload fetch, cache
  `{id: {source_file, caption}}` for the current turn's fused pool during
  `/api/search` - the payloads are already in memory there. Zero extra calls,
  and the cache is naturally bounded (overwritten every search, size =
  `SUBMISSION_TOP_K`).
- Order inside the type-1 block matters: **boost first, then ambiguity, then
  rerank**. Boosting sharpens the top-1/top-2 margin, so a clarification that
  actually worked will *lower* the new ambiguity score and stop the system asking
  a second redundant question - which is exactly the behaviour we want on a
  7-minute clock.
- `pending_clarification` must be consumed (set to `None`) on **every**
  `/api/search`, including type 2/3 turns, so a stale flag can never boost a
  much later unrelated turn.

## Requirements

### Functional

1. `_session_state` gains `pending_clarification` and `last_candidate_info`.
2. `SearchRequest` gains `clarification_answer: Optional[str] = None`.
3. `/api/search` type 1: when a clarification was asked last turn **and** this
   request carries a `clarification_answer`, apply
   `boost_by_clarification_answer` to the fused pool before the ambiguity check
   and `rerank_type1`.
4. When a clarification question is generated, record the question and the ids of
   the (up to 5) distinct-video candidates it was built from in
   `pending_clarification`.
5. `/api/feedback` annotates the latest history turn with descriptions of the
   accepted/rejected candidates, so the next `rewrite_query_cqr` sees them.
6. Response payload optionally reports the boost was applied (see step 6) for
   operator transparency; existing response keys unchanged.
7. Frontend sends `clarification_answer` alongside the existing composed query.

### Non-functional

- No change to the number of LLM/VLM calls per turn, in any branch.
- No new Qdrant round-trip.
- Backwards compatible: a client that never sends `clarification_answer` behaves
  exactly as today; `/api/feedback` with no prior search still 400s as today.
- `main.py` is already 1176 lines (over the 200-line convention) - this phase must
  keep its additions small and put all real logic in the phase 01/02 pure
  modules, not here.

## Architecture

`/api/search` type-1 block, after `diversify_by_scene`:

```
candidates (RRF-fused, temporally boosted, diversified)
   |
   | pending = session["pending_clarification"];  session["pending_clarification"] = None
   |
   +-- if type==1 and pending and request.clarification_answer:
   |        candidates = boost_by_clarification_answer(
   |            candidates, pending["candidate_ids"], request.clarification_answer)
   |
   +-- ambiguity = searcher.compute_ambiguity_score(candidates)      # phase 01 combined signal
   |
   +-- if ambiguity >= AMBIGUITY_THRESHOLD:
   |        summaries, summary_ids = <existing distinct-video loop, now also collecting ids>
   |        clarification_question = query_proc.generate_clarification_question(...)
   |        session["pending_clarification"] = {"question": ..., "candidate_ids": summary_ids}
   |
   +-- reranker.rerank_type1(...)
```

`/api/feedback`:

```
positive_ids / negative_ids
   -> record_feedback_in_history(session["history"], session["last_candidate_info"],
                                 positive_ids, negative_ids)
   -> history[-1]["accepted"/"rejected"]  ->  read by the NEXT build_cqr_prompt()
   (Rocchio vector adjustment + re-search unchanged)
```

## Related Code Files

**Modify**

- `webapp/backend/main.py`
- `webapp/frontend/src/App.tsx` (one added line)

**Create / Delete** - none.

## Implementation Steps

1. `_session_state` (L89) - add two keys and extend the existing comment block:
   ```python
   _session_state = {
       "history": [],
       "last_query_vector": None,
       # KIS-C clarification round-trip: set when the previous turn asked a
       # clarifying question; holds that question plus the candidate ids it was
       # generated from, so the next turn's answer can boost exactly those
       # candidates (Sekulic et al. arXiv:2008.03717). Consumed (reset to None)
       # by every /api/search.
       "pending_clarification": None,
       # {point_id: {"source_file", "caption"}} for the last search's fused pool,
       # so /api/feedback can describe the operator's accepted/rejected picks in
       # words for the next CQR rewrite without a Qdrant payload fetch.
       "last_candidate_info": {},
   }
   ```

2. `SearchRequest` - add:
   ```python
   # KIS-C: the operator's answer to the clarifying question asked last turn,
   # sent separately from `query` (which still carries it appended, for
   # retrieval) so the backend can boost the exact candidates the question was
   # about without string-parsing the composed query.
   clarification_answer: Optional[str] = None
   ```

3. In `run_search`, right after `diversify_by_scene` / the `last_query_vector` +
   `history.append` block, populate the payload cache from the in-memory pool:
   ```python
   _session_state["last_candidate_info"] = {
       c["id"]: {
           "source_file": (c.get("payload") or {}).get("source_file"),
           "caption": ((c.get("payload") or {}).get("caption") or "")[:200],
       }
       for c in candidates
   }
   ```
   Note `history.append({"query": resolved_query})` already happened, so
   `history[-1]` is this turn - which is what `/api/feedback` will annotate.

4. Consume the pending flag and apply the boost. Put this at the top of the
   `if request.type == 1:` block, before `compute_ambiguity_score`:
   ```python
   from search.kis_c_scoring import boost_by_clarification_answer
   pending = _session_state["pending_clarification"]
   _session_state["pending_clarification"] = None
   clarification_boost_applied = False
   if pending and (request.clarification_answer or "").strip():
       candidates = boost_by_clarification_answer(
           candidates, pending.get("candidate_ids") or [], request.clarification_answer
       )
       clarification_boost_applied = True
   ```
   Local import matches the file's existing style (`import config`,
   `from search.query_processor import ...` inside functions).
   **Also** reset `pending_clarification` for the type 2/3 paths - simplest
   correct form: do the `pending = ...; _session_state["pending_clarification"] = None`
   pair once *before* the `if request.type == 1:` branch, and keep only the
   boost call inside the type-1 branch.
   Add a comment stating this is an additive boost on the normal per-turn pool,
   deliberately **not** a fast path that skips search.

5. Extend the existing clarification block to also collect ids and set the flag:
   ```python
   summary_ids = []
   ...
   summaries.append(...); summary_ids.append(c["id"])
   ...
   clarification_question = query_proc.generate_clarification_question(resolved_query, summaries)
   _session_state["pending_clarification"] = {
       "question": clarification_question,
       "candidate_ids": summary_ids,
   }
   ```
   Keep the existing `AMBIGUITY_THRESHOLD` gate untouched (phase 01 preserved the
   float return).

6. Response: add `"clarification_boost_applied": clarification_boost_applied` to
   the `/api/search` return dict (default `False` for type 2/3). Additive key
   only - the frontend ignores unknown keys, so no UI work is required. Justified
   by the same explainability rationale as `matched_via`.

7. `/api/feedback` - after the Rocchio adjustment succeeds, before/after the
   re-search (order irrelevant, keep it next to the state writes):
   ```python
   from search.conversational_context import record_feedback_in_history
   # Exquisitor-inspired (VBS 2024/2025) unified conversational + relevance
   # feedback loop, in its lightest prompt-only form: the operator's
   # accept/reject signal was previously isolated in the Rocchio vector and
   # never reached CQR. Recording it on the current history turn means the
   # NEXT rewrite_query_cqr prompt knows which readings were already rejected.
   # Same single CQR call as before - prompt context only.
   record_feedback_in_history(
       _session_state["history"], _session_state["last_candidate_info"],
       request.positive_ids, request.negative_ids,
   )
   ```
   No change to `FeedbackRequest`, the Rocchio math, or the response shape.

8. `webapp/frontend/src/App.tsx` - in the non-temporal request body (L248-253) add:
   ```ts
   ...(isConversationalTask && clarificationAnswer.trim()
     ? { clarification_answer: clarificationAnswer.trim() }
     : {}),
   ```
   `requestQuery` composition stays exactly as-is (the answer still reaches CQR
   and retrieval through `query`). No other frontend change.

9. Compile + smoke check:
   - `python3 -m py_compile webapp/backend/main.py`
   - frontend typecheck/build (`npm run build` in `webapp/frontend`, or the
     project's existing script) - the added spread is plain TS, no type change.
   - Manual: start the backend, run a deliberately vague type-1 query, confirm a
     clarification comes back, answer it, confirm the second response has
     `clarification_boost_applied: true` and that no second question is asked
     when the answer was discriminating.

## Todo List

- [x] `_session_state`: `pending_clarification`, `last_candidate_info` (+ comments)
- [x] `SearchRequest.clarification_answer`
- [x] Populate `last_candidate_info` from the fused pool
- [x] Consume `pending_clarification` once per search (all types)
- [x] Apply `boost_by_clarification_answer` before the ambiguity check (type 1)
- [x] Collect `summary_ids` + set `pending_clarification` when a question is asked
- [x] Add `clarification_boost_applied` to the response
- [x] `/api/feedback` -> `record_feedback_in_history`
- [x] `App.tsx` sends `clarification_answer`
- [x] `py_compile` main.py + frontend build clean
- [x] Manual round-trip smoke check

## Success Criteria

- LLM/VLM call count per turn identical to before: type 1 = CQR + HyDE + (0|1)
  clarification + rerank. Verify by grepping the diff for `vlm`/`generate` -
  there must be no new call site.
- No new `client.retrieve` / `search` / `scroll` call added.
- First turn of a session behaves byte-for-byte as today (no pending flag, no
  boost, `last_candidate_info` merely populated).
- After answering a clarification, `clarification_boost_applied` is `true` and the
  candidate the answer describes ranks above its pre-answer position.
- `pending_clarification` is `None` after any `/api/search`, unless that search
  itself asked a new question.
- `/api/feedback` before any search still returns the same 400.
- After `/api/feedback`, `_session_state["history"][-1]` contains
  `accepted`/`rejected` descriptions and the next `/api/search` CQR prompt
  includes them.
- Existing callers (CLI `main.py`, `batch_query.py`, `evaluation/run_eval.py`) are
  untouched and unaffected.

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Stale `pending_clarification` boosts an unrelated later turn | Consumed unconditionally on every `/api/search`, before the type branch |
| Operator answers the clarification but the retained ids no longer appear in the new turn's pool -> boost is a silent no-op | Expected and safe (the query mostly repeats, so overlap is high). `clarification_boost_applied` plus phase 01's `clarification_overlap` field make it visible instead of silent |
| Frontend and backend disagree (answer sent in `query` only) | Backend guards on `(request.clarification_answer or "").strip()`; without the field the code path is simply skipped - no crash, just today's behaviour |
| Global single-session state races between concurrent requests | Pre-existing "one operator per backend instance" design (documented at L84-88); this phase adds no new sharing model. Do not introduce locking here |
| `last_candidate_info` grows unbounded | Rebuilt (not appended) each search; size = `SUBMISSION_TOP_K` |
| `main.py` grows further past the file-size convention | All logic lives in the phase 01/02 pure modules; this phase adds ~30 lines of wiring. Flag a future `main.py` split as follow-up, do not attempt it here |

## Security Considerations

- `clarification_answer` is operator-supplied text used only for tokenization
  (phase 01) and prompt text (phase 02) - never in a Qdrant filter, path, shell
  command, or `eval`. FastAPI/Pydantic validates it as an optional string.
- No new endpoint, no auth change; the KIS-C session remains the existing
  single-operator trusted local deployment.
- `last_candidate_info` holds only our own index metadata (video filename +
  caption) in process memory; nothing is persisted or logged beyond the existing
  `interaction_log` calls.

## Next Steps

- Phase 04: unit tests for the pure modules + manual verification checklist + CHANGELOG.
- Follow-up (out of scope): tune `AMBIGUITY_THRESHOLD` and `MARGIN_WEIGHT` against
  recorded VBS-2025 KIS-C tasks (researcher-a §6 - a wrong guess costs 10 pts ~ 60s).

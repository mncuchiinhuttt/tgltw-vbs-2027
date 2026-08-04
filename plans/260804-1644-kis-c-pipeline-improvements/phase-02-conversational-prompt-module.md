# Phase 02 - Conversational Prompt Module (few-shot CQR + facet clarification + feedback context)

## Context Links

- Research: `../260804-0415-kis-c-conversational-retrieval-research/research/researcher-b-conversational-rewriting-sota.md` (§2 PG-ICL, arXiv:2502.15009)
- Research: `../260804-0415-kis-c-conversational-retrieval-research/research/researcher-c-clarifying-question-sota.md` (§2 facet-driven clarification, Sekulic et al. SIGIR-TIIR 2021 / zero-shot variant arXiv:2301.12660; referring-expression-generation disambiguation)
- Research: `../260804-0415-kis-c-conversational-retrieval-research/research/researcher-a-vbs-competitor-techniques.md` (technique #1, Exquisitor VBS 2024/2025 unified conversational + relevance-feedback loop)
- Code: `inference-code/search/query_processor.py` (`rewrite_query_cqr` L12, `generate_clarification_question` L81)
- Code: `webapp/backend/main.py` L86-89 (session `history` shape), L488 (only VQA turns get `answer`)
- Plan overview: `plan.md`

## Overview

- **Priority:** P2 (blocks phase 03)
- **Status:** done
- **Effort:** ~1.5h
- Scope items **#1** (few-shot CQR prompt), **#2** (facet-driven clarifying
  question), and the **prompt half of #5** (Rocchio accept/reject signal as CQR
  context). All three are prompt engineering inside the *existing* single LLM
  call per function. Call count unchanged.

## Key Insights

- `rewrite_query_cqr` is zero-shot today; PG-ICL (arXiv:2502.15009) shows a
  few-shot template with a fixed structured output format is the single cheapest
  win available - same one call, richer prompt.
- `generate_clarification_question` today dumps up to 5 captions and asks for
  "ONE short, specific clarifying question" with no guidance on *what to ask
  about*, so it drifts to generic questions. The facet-driven fix is
  chain-of-thought **inside the same call**: first name the attribute that
  actually differs across the given captions, then ask about exactly that. The
  question is what the operator relays verbally to the moderator, so a question
  about a non-differentiating facet burns ~10s of a 7-minute clock for nothing.
- The instruction must constrain the facet to **what is visible in the caption
  text** - the LLM sees no frames here, only caption strings.
- Queries are Vietnamese *or* English (see the existing "Vietnamese is OK" in
  `generate_hyde` and the Vietnamese examples in `decompose_query`), so the
  few-shot examples must cover both, and the output-language rule ("answer in the
  same language as the latest query") must be explicit.
- Prompt-only changes are normally untestable in CI. Moving the strings into a
  pure builder module converts them into **testable structure** (placeholders
  present, example count, facet instruction present, feedback lines rendered) -
  this is the main reason for the extra module, not aesthetics.
- `query_processor.py` is 135 lines; inlining ~60 lines of few-shot examples +
  facet instructions would break the repo's 200-line file convention.
- Exquisitor's unified loop is implemented here in its **lightest form only**:
  the accept/reject signal becomes extra prompt context. No unified UI, no
  re-ranking from feedback, no new call.

## Requirements

### Functional

1. `build_cqr_prompt(query, context_history) -> str` produces a few-shot prompt
   with 3 static examples: (a) English pronoun resolution, (b) Vietnamese
   implicit reference, (c) a turn carrying rejected-candidate feedback.
2. `format_history(context_history) -> str` renders each turn as the existing
   `User:` / `System:` lines **plus** optional `Operator rejected:` /
   `Operator confirmed:` lines when that turn carries feedback.
3. `build_clarification_prompt(query, candidate_summaries) -> str` instructs the
   model to (a) identify the differing facet, (b) ask ONE question about that
   facet, (c) output only the question.
4. `record_feedback_in_history(history, candidate_info, positive_ids, negative_ids, limit=3) -> None`
   attaches short human-readable descriptions of accepted/rejected candidates to
   the most recent history turn.
5. `describe_candidates(candidate_info, ids, limit=3) -> list[str]` maps point ids
   to short caption/video descriptions.
6. `QueryProcessor.rewrite_query_cqr` and `generate_clarification_question` keep
   their exact signatures, return types, existing early-return behaviour
   (`rewrite_query_cqr` returns `query` unchanged when history is empty) and
   existing debug prints. Each still makes **exactly one** `self.vlm.generate`
   call.

### Non-functional

- New module: stdlib only, no imports from `config`/`models`/`qdrant_client`.
- Under 200 lines including the example block.
- All builders total: `None`/empty inputs produce a valid prompt, never raise.

## Architecture

```
webapp/backend/main.py
  /api/search   -> QueryProcessor.rewrite_query_cqr(query, history)
                        -> conversational_context.build_cqr_prompt()   [pure]
                        -> self.vlm.generate(None, prompt)             [1 call, as today]
                -> QueryProcessor.generate_clarification_question(q, summaries)
                        -> conversational_context.build_clarification_prompt()  [pure]
                        -> self.vlm.generate(None, prompt)             [1 call, as today]
  /api/feedback -> conversational_context.record_feedback_in_history(...)  [pure, 0 calls]
                        writes history[-1]["rejected"] / ["accepted"]
                        which format_history() renders into the NEXT rewrite
```

Data flow for #5 (the loop that is closed by this phase + phase 03):

```
/api/search  -> session_state["last_candidate_info"] = {id: {source_file, caption}}
/api/feedback(positive_ids, negative_ids)
             -> describe_candidates(...) -> history[-1]["accepted"/"rejected"]
next /api/search -> build_cqr_prompt() renders those lines -> better rewrite
```

## Related Code Files

**Create**

- `inference-code/search/conversational_context.py`

**Modify**

- `inference-code/search/query_processor.py` - `rewrite_query_cqr`, `generate_clarification_question` (bodies only).

**Delete** - none.

## Implementation Steps

1. Create `inference-code/search/conversational_context.py`. Module docstring:
   what it holds (KIS-C conversational prompt construction + session history
   annotation), why it is separate (pure/testable + keeps `query_processor.py`
   under 200 lines), and the hard rule **"no function here may add an LLM call -
   these build strings for calls that already exist"**.

2. `CQR_FEWSHOT_EXAMPLES` - a module-level triple-quoted constant with 3 examples
   in a fixed `History: / Latest Query: / Rewritten Query:` shape matching the
   live section, e.g.:
   - EN pronoun: history `User: a man in a red jacket walking a dog in a park` /
     latest `what is he holding` -> `what is the man in the red jacket walking a
     dog in a park holding`.
   - VI implicit reference: history `User: hai người đàn ông đang sửa xe máy bên
     đường` / latest `cảnh đó có biển hiệu không` -> `hai người đàn ông đang sửa
     xe máy bên đường, trong cảnh có biển hiệu`.
   - Feedback-carrying: history includes an `Operator rejected: a woman cooking
     in a kitchen` line and the rewrite steers away from that reading while
     keeping the original intent.
   Keep each example 3-5 lines. Do NOT let an example invent facts absent from
   the history - the examples teach the behaviour we want.

3. `CQR_INSTRUCTIONS` - task description + rules:
   - resolve pronouns and implicit references using the history;
   - keep it one concise self-contained descriptive sentence;
   - **reply in the same language as the latest query**;
   - if `Operator rejected:` lines are present, avoid re-describing those
     interpretations while preserving the user's intent; `Operator confirmed:`
     lines are partial matches worth keeping;
   - output ONLY the rewritten query, no preamble.

4. `def format_history(context_history) -> str`:
   - `for turn in context_history or []`: `User: {turn.get("query","")}` then
     `System: {turn["answer"]}` only when non-empty (today only VQA turns have
     one - do not emit an empty `System:` line, it teaches the model that
     answers are usually blank);
   - `Operator rejected: a; b; c` when `turn.get("rejected")`;
   - `Operator confirmed: a; b` when `turn.get("accepted")`.
   - Join with `\n`.

5. `def build_cqr_prompt(query, context_history) -> str` - assemble
   `CQR_INSTRUCTIONS` + `CQR_FEWSHOT_EXAMPLES` + a `Now do the same:` separator +
   the live `History:` / `Latest Query:` / `Rewritten Query:` block. The live
   block must be byte-identical in shape to the examples' block.

6. `def build_clarification_prompt(query, candidate_summaries) -> str`. Prompt
   body (single call, reasoning folded in):
   - context: these are the top results for an ambiguous video search, each from
     a different video;
   - numbered candidate captions;
   - `Original query: "<query>"`;
   - **Step 1 (internal):** decide which single attribute actually DIFFERS across
     the captions above - e.g. object colour, clothing, action, location/setting,
     number of people, on-screen text - considering only what the caption text
     states;
   - **Step 2:** write ONE short question asking specifically about that
     differing attribute, phrased so that any answer eliminates at least one
     candidate; do not ask about anything all candidates share, and do not ask a
     generic "what happens in the clip" question;
   - reply in the same language as the original query;
   - output ONLY the question - do not output the chosen attribute or the
     reasoning.
   - Keep the docstring citation: Sekulic et al. facet-driven clarification
     (zero-shot variant) + referring-expression-generation disambiguation.

7. `def describe_candidates(candidate_info, ids, limit=3) -> list[str]`:
   - for each id in `ids or []`, look up `(candidate_info or {}).get(id)`;
   - description = truncated `caption` (<=120 chars) or else `source_file` or else
     the id itself (never drop the signal entirely);
   - de-duplicate, preserve order, cap at `limit` (3 keeps the prompt short - the
     operator may mark 20 results).

8. `def record_feedback_in_history(history, candidate_info, positive_ids, negative_ids, limit=3) -> None`:
   - no-op when `history` is empty (feedback before any search);
   - `turn = history[-1]`; extend `turn.setdefault("accepted", [])` /
     `turn.setdefault("rejected", [])` with `describe_candidates(...)`,
     de-duplicating and re-capping at `limit` (an operator can hit
     `/api/feedback` several times on one turn).
   - Mutates in place, returns `None`; docstring must say so.

9. Edit `query_processor.py`:
   - add `from search.conversational_context import build_cqr_prompt, build_clarification_prompt`
     at the top (`sys.path` already contains `inference-code` by the time this is
     imported - `hybrid_search.py` does the same style of intra-`search` import
     work; if an import-order problem shows up, do the import inside the method
     rather than adding another `sys.path` hack).
   - `rewrite_query_cqr`: keep the `if not context_history: return query` guard,
     replace the inline f-string with `prompt = build_cqr_prompt(query, context_history)`,
     keep the single `self.vlm.generate(None, prompt).strip()` and the existing
     `print`. Extend the docstring: few-shot (PG-ICL arXiv:2502.15009), and that
     `history` turns may now carry `accepted`/`rejected` feedback descriptions
     (Exquisitor-inspired, prompt-only).
   - `generate_clarification_question`: replace the inline prompt with
     `build_clarification_prompt(query, candidate_summaries)`; keep the existing
     KIS-C docstring and append the facet-driven rationale.
   - Net effect: `query_processor.py` gets *shorter*.

10. Compile check: `python3 -m py_compile inference-code/search/conversational_context.py inference-code/search/query_processor.py`.

## Todo List

- [x] Create `conversational_context.py` (docstring + no-extra-call rule)
- [x] `CQR_INSTRUCTIONS` + `CQR_FEWSHOT_EXAMPLES` (EN pronoun, VI implicit, feedback-carrying)
- [x] `format_history` (answer / rejected / confirmed lines)
- [x] `build_cqr_prompt`
- [x] `build_clarification_prompt` (facet identification -> one question -> question only)
- [x] `describe_candidates`
- [x] `record_feedback_in_history`
- [x] `query_processor.py` delegates both prompts; signatures + call counts unchanged
- [x] `py_compile` clean

## Success Criteria

- `grep -c "vlm.generate" inference-code/search/query_processor.py` is unchanged from before the edit.
- `rewrite_query_cqr("x", [])` returns `"x"` without touching the VLM (unchanged).
- `build_cqr_prompt` output contains: the latest query, every history query, >=3
  `Rewritten Query:` occurrences (3 examples + the live slot), and any
  `rejected` description present in the history.
- `build_clarification_prompt` output contains every supplied summary, the query,
  and an explicit "differs"/"only the question" instruction.
- `record_feedback_in_history([], ...)` is a no-op; called twice with different
  ids, the descriptions accumulate without duplicates and stay <= `limit`.
- `conversational_context.py` and `query_processor.py` both under 200 lines.

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Few-shot examples bias the model toward the example domains (dogs, motorbikes) | Keep examples short, structurally-focused, and mixed-language; phase 04 manual checklist eyeballs real rewrites on 3+ live queries |
| Longer prompt = higher latency / token cost | 3 short examples are tens of tokens; still one call. Measure in phase 04's manual check and drop to 2 examples if the rewrite call visibly slows |
| The model outputs its facet reasoning instead of just the question | Explicit "output ONLY the question" + phase 04 manual check; the frontend renders the raw string into the amber banner, so a leaked preamble is immediately visible |
| A "rejected" description in the prompt makes the model over-correct and lose the original intent | Instruction says "avoid re-describing those interpretations *while preserving the user's intent*"; example (c) demonstrates it; capped at 3 descriptions |
| Truncated captions in feedback lines lose the discriminating detail | 120 chars covers typical caption length; the value is directional ("not the kitchen one"), not exact |
| Prompt behaviour cannot be verified in CI | Accepted and documented: structural tests only (phase 04), plus an explicit manual verification checklist. Do not fake an LLM to claim coverage |

## Security Considerations

- Operator/moderator free text (query, clarification answer, captions) is
  interpolated into an LLM prompt - prompt-injection is theoretically possible
  ("ignore previous instructions"). Impact is bounded: the output is a search
  string / a question shown to the operator, never executed, never used to build
  a filesystem path or shell command. No change to the existing exposure level.
- Captions in `describe_candidates` come from our own index, not from users.
- No credentials, no new network surface, no new dependency.

## Next Steps

- Phase 03 calls `record_feedback_in_history` from `/api/feedback` and populates `last_candidate_info` in `/api/search`.
- Phase 04 adds `tests/test_conversational_context.py` + the manual verification checklist.

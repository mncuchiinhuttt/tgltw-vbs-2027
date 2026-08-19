"""
KIS-C conversational prompt construction + session-history annotation.

Pure string/dict logic - no imports at all, so it is safe to import from
anywhere. Kept separate from query_processor.py so the prompt templates are
structurally unit-testable without an LLM, and so query_processor.py stays
under this repo's 200-line-file convention.

Hard rule: no function here may add an LLM call - these only build strings
for calls that already exist in query_processor.py.
"""

# Few-shot examples for CQR (PG-ICL, arXiv:2502.15009): fixed
# History: / Latest Query: / Rewritten Query: shape: one per language, one
# demonstrating rejected-feedback handling, and one demonstrating a KIS-C
# clarification round-trip. Keep short - these ride along on every rewrite call.
CQR_FEWSHOT_EXAMPLES = """Example 1:
History:
User: a man in a red jacket walking a dog in a park
Latest Query: what is he holding
Rewritten Query: what is the man in the red jacket walking a dog in the park holding

Example 2:
History:
User: hai người đàn ông đang sửa xe máy bên đường
Latest Query: cảnh đó có biển hiệu không
Rewritten Query: hai người đàn ông đang sửa xe máy bên đường, trong cảnh có biển hiệu

Example 3:
History:
User: a woman cooking in a kitchen
Operator rejected: a woman cooking in a kitchen
Latest Query: no, it's outdoors, she's grilling
Rewritten Query: a woman grilling food outdoors

Example 4:
History:
User: một người phụ nữ đang nấu ăn trong bếp
System asked: người phụ nữ đó mặc áo màu gì?
Latest Query: một người phụ nữ đang nấu ăn trong bếp. Additional detail from operator: áo màu vàng
Rewritten Query: một người phụ nữ mặc áo màu vàng đang nấu ăn trong bếp"""

# The negation clause is load-bearing, not a stylistic preference: this rewrite
# is what gets embedded, HyDE'd and handed to rerank_type1, and NevIR (Weller
# et al., EACL 2024, arXiv:2305.07614) finds bi-encoder and sparse retrievers rank negated pairs
# WORSE than random. Converting "not red, it's blue" into a purely affirmative
# description is the only point in the pipeline where that can be repaired
# before it reaches a retriever that cannot represent it.
CQR_INSTRUCTIONS = """You are a Query Rewriter. Given the conversation history and the latest user query, rewrite the latest query to be fully self-contained and descriptive, resolving any pronouns or implicit references. Keep it one concise descriptive sentence. Reply in the same language as the latest query. If "Operator rejected:" lines are present, avoid re-describing those interpretations while preserving the user's original intent; "Operator confirmed:" lines are partial matches worth keeping. If the latest query denies or corrects a detail ("not red", "không phải màu đỏ"), state what the target IS and never carry the denied attribute into the rewritten query - write the positive description only. Output ONLY the rewritten query, no preamble."""

def format_history(context_history: list) -> str:
    """
    Renders each turn as `User:` / `System:` lines, plus optional
    `Operator rejected:` / `Operator confirmed:` lines when that turn
    carries feedback (see record_feedback_in_history), plus a final
    `System asked:` line when the turn ended by asking a KIS-C clarifying
    question. Optional lines are omitted when absent rather than emitted
    blank - a blank line would teach the model that field is usually empty.

    `System asked:` comes LAST because it is the utterance the NEXT user turn
    answers; without it the rewriter sees a bare "áo màu vàng" and must guess
    which attribute was narrowed. Kept distinct from `System:` (a VQA answer)
    so the two roles are not conflated.
    """
    lines = []
    for turn in context_history or []:
        lines.append(f"User: {turn.get('query', '')}")
        if turn.get("answer"):
            lines.append(f"System: {turn['answer']}")
        if turn.get("rejected"):
            lines.append(f"Operator rejected: {'; '.join(turn['rejected'])}")
        if turn.get("accepted"):
            lines.append(f"Operator confirmed: {'; '.join(turn['accepted'])}")
        if turn.get("clarification_asked"):
            lines.append(f"System asked: {turn['clarification_asked']}")
    return "\n".join(lines)


def build_cqr_prompt(query: str, context_history: list) -> str:
    """
    Few-shot CQR prompt (PG-ICL, arXiv:2502.15009): task instructions + 4
    static examples + the live History:/Latest Query:/Rewritten Query: block,
    kept byte-identical in shape to the examples so the model pattern-matches
    the live slot the same way.
    """
    history_str = format_history(context_history)
    return f"""{CQR_INSTRUCTIONS}

{CQR_FEWSHOT_EXAMPLES}

Now do the same:
History:
{history_str}
Latest Query: {query}
Rewritten Query:"""


def build_clarification_prompt(query: str, candidate_summaries: list) -> str:
    """
    Facet-driven clarifying question (Sekulic et al., SIGIR-TIIR 2021
    facet-driven clarification, zero-shot variant arXiv:2301.12660; also
    referring-expression-generation disambiguation): a single LLM call that
    folds two steps into one - (1) identify which attribute actually DIFFERS
    across the given candidate captions, considering only what the caption
    text states (colour, clothing, action, location/setting, number of
    people, on-screen text, etc.), then (2) ask ONE short question about
    exactly that differing attribute, phrased so any answer eliminates at
    least one candidate. Never asks about an attribute all candidates share,
    never a generic "what happens in the clip" question. Output must be ONLY
    the question - no leaked reasoning/preamble, since the raw string is
    shown directly to the operator.
    """
    summaries_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(candidate_summaries or []))
    return f"""You are helping refine an ambiguous video search query. Below are the top results, each from a different video:
{summaries_str}
Original query: "{query}"
Step 1 (internal, do not output): decide which single attribute actually DIFFERS across the captions above.
Step 2: write ONE short question asking specifically about that differing attribute, phrased so that any answer eliminates at least one candidate. Do not ask about anything all candidates share, and do not ask a generic "what happens in the clip" question.
Reply in the same language as the original query. Output ONLY the question - do not output the chosen attribute or your reasoning.
Question:"""


def describe_candidates(candidate_info: dict, ids: list, limit: int = 3) -> list:
    """
    Maps point ids to short human-readable descriptions (truncated caption,
    else source_file, else the raw id - never drop the signal entirely).
    De-duplicates, preserves order, caps at `limit` (an operator can mark 20
    results, but the CQR prompt should stay short).
    """
    seen = set()
    descriptions = []
    for cid in ids or []:
        info = (candidate_info or {}).get(cid) or {}
        desc = (info.get("caption") or "")[:120] or info.get("source_file") or str(cid)
        if desc in seen:
            continue
        seen.add(desc)
        descriptions.append(desc)
        if len(descriptions) >= limit:
            break
    return descriptions


def record_feedback_in_history(history: list, candidate_info: dict, positive_ids: list, negative_ids: list, limit: int = 3) -> None:
    """
    Exquisitor-inspired (VBS 2024/2025 unified conversational + relevance-
    feedback loop, lightest prompt-only form): attaches short descriptions of
    accepted/rejected candidates to the most recent history turn, so the
    NEXT build_cqr_prompt renders them as "Operator rejected:"/"Operator
    confirmed:" lines. No-op when `history` is empty (feedback given before
    any search). Mutates `history` in place, returns None. An operator can
    call /api/feedback more than once per turn - descriptions accumulate
    without duplicates and stay capped at `limit`.
    """
    if not history:
        return
    turn = history[-1]
    accepted = turn.setdefault("accepted", [])
    rejected = turn.setdefault("rejected", [])
    for desc in describe_candidates(candidate_info, positive_ids, limit):
        if desc not in accepted:
            accepted.append(desc)
    for desc in describe_candidates(candidate_info, negative_ids, limit):
        if desc not in rejected:
            rejected.append(desc)
    del accepted[limit:]
    del rejected[limit:]

"""
Per-task session-state maintenance for the KIS-C conversational loop.

Pure dict/list logic, no imports - the counterpart to conversational_context.py
(which RENDERS the session into prompts). Split out so both stay under this
repo's 200-line-file convention and so each has one job: this module decides
what the session remembers, that one decides how it is shown to the model.
"""


# Upper bound on how many turns reach build_cqr_prompt: every turn is rendered
# verbatim, so an untrimmed history grows the prompt and the rewrite latency
# without bound on a 7-minute clock. 8 is well above what one task reaches.
MAX_HISTORY_TURNS = 8


def trim_history(history: list, max_turns: int = MAX_HISTORY_TURNS) -> None:
    """
    Caps `history` at `max_turns` entries in place, keeping the FIRST turn plus
    the most recent ones. Turn 1 is preserved deliberately: a KIS-C task opens
    with the oracle panel's initial description (VBS_GUIDE.md section 4.1) and
    every later turn only adds detail to it, so dropping it would discard the
    anchor the whole conversation refines. Mutates in place (like
    record_feedback_in_history) so holders of the same list see the trim.
    """
    if max_turns <= 0 or len(history) <= max_turns:
        return
    # Slice the tail by absolute index, not by -(max_turns - 1): at
    # max_turns == 1 that negates to history[-0:], which is the WHOLE list
    # rather than an empty one, and the trim silently does nothing.
    tail = history[len(history) - (max_turns - 1):] if max_turns > 1 else []
    history[:] = history[:1] + tail


def build_candidate_info(results: list, caption_chars: int = 200) -> dict:
    """
    Builds the {point_id: {"source_file", "caption"}} cache describe_candidates
    reads, from whatever result set is currently on the operator's screen.

    Every endpoint that REPLACES the displayed results must refresh this, not
    just /api/search: the next accept/reject names ids from what the operator
    can see, and an id missing here makes describe_candidates fall through to
    str(cid), injecting a raw point UUID into the CQR prompt as an "Operator
    rejected:" line. Entries without an id (temporal-search's per-chain rows)
    are skipped rather than keyed on None.
    """
    info = {}
    for hit in results or []:
        cid = (hit or {}).get("id")
        if cid is None:
            continue
        payload = hit.get("payload") or {}
        info[cid] = {
            "source_file": payload.get("source_file"),
            "caption": (payload.get("caption") or "")[:caption_chars],
        }
    return info

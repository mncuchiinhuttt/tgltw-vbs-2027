"""
Unit tests for inference-code/search/conversational_context.py (KIS-C
few-shot CQR prompt, facet-driven clarification prompt, history formatting,
feedback recording).

Pure logic test - stub dicts only, no LLM / VLM / network access. What is
NOT covered here (documented limitation, not silently skipped): whether the
LLM actually produces a better rewrite/question from these prompts - that
requires the manual verification checklist in phase-04's plan doc, executed
against a live VLM.

Runnable both under pytest and as a plain script:
    python tests/test_conversational_context.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.conversational_context import (
    format_history,
    build_cqr_prompt,
    build_clarification_prompt,
    describe_candidates,
    record_feedback_in_history,
)


# --- format_history / build_cqr_prompt ---

def test_build_cqr_prompt_contains_latest_query_and_history():
    history = [{"query": "a man in a red jacket"}]
    prompt = build_cqr_prompt("what is he holding", history)
    assert "a man in a red jacket" in prompt
    assert "what is he holding" in prompt


def test_build_cqr_prompt_has_at_least_three_examples():
    prompt = build_cqr_prompt("q", [{"query": "prev"}])
    assert prompt.count("Rewritten Query:") >= 4


def test_build_cqr_prompt_renders_rejected_and_accepted_lines():
    history = [{"query": "a woman cooking", "rejected": ["a woman cooking in a kitchen"], "accepted": ["a woman outdoors"]}]
    prompt = build_cqr_prompt("no, outdoors", history)
    assert "Operator rejected: a woman cooking in a kitchen" in prompt
    assert "Operator confirmed: a woman outdoors" in prompt


def test_build_cqr_prompt_omits_empty_system_answer_line():
    history = [{"query": "a man walking"}]
    prompt = build_cqr_prompt("q", history)
    assert "System:" not in prompt


def test_build_cqr_prompt_empty_history_does_not_raise():
    # rewrite_query_cqr short-circuits on empty history before ever calling
    # build_cqr_prompt, but the builder itself must still be total.
    prompt = build_cqr_prompt("q", [])
    assert "Latest Query: q" in prompt


def test_format_history_handles_none():
    assert format_history(None) == ""


def test_format_history_includes_answer_when_present():
    history = [{"query": "what color is the car", "answer": "red"}]
    rendered = format_history(history)
    assert "User: what color is the car" in rendered
    assert "System: red" in rendered


# --- build_clarification_prompt ---

def test_build_clarification_prompt_lists_all_summaries():
    summaries = ["a man in a red jacket", "a woman in a blue coat"]
    prompt = build_clarification_prompt("someone walking", summaries)
    assert summaries[0] in prompt
    assert summaries[1] in prompt
    assert "someone walking" in prompt


def test_build_clarification_prompt_instructs_facet_identification():
    prompt = build_clarification_prompt("q", ["a", "b"]).lower()
    assert "differ" in prompt
    assert "only the question" in prompt


def test_build_clarification_prompt_empty_summaries_does_not_raise():
    prompt = build_clarification_prompt("q", [])
    assert "q" in prompt


# --- describe_candidates ---

def test_describe_candidates_prefers_caption_then_video_then_id():
    info = {
        "a": {"caption": "a red car", "source_file": "V001.mp4"},
        "b": {"caption": "", "source_file": "V002.mp4"},
        "c": {},
    }
    assert describe_candidates(info, ["a"]) == ["a red car"]
    assert describe_candidates(info, ["b"]) == ["V002.mp4"]
    assert describe_candidates(info, ["c"]) == ["c"]


def test_describe_candidates_truncates_and_caps_at_limit():
    long_caption = "x" * 300
    info = {str(i): {"caption": long_caption if i == 0 else f"caption {i}"} for i in range(5)}
    result = describe_candidates(info, [str(i) for i in range(5)], limit=3)
    assert len(result) == 3
    assert len(result[0]) <= 120


def test_describe_candidates_deduplicates():
    info = {"a": {"caption": "same caption"}, "b": {"caption": "same caption"}}
    result = describe_candidates(info, ["a", "b"], limit=5)
    assert result == ["same caption"]


# --- record_feedback_in_history ---

def test_record_feedback_in_history_no_history_is_noop():
    history = []
    record_feedback_in_history(history, {}, ["a"], ["b"])
    assert history == []


def test_record_feedback_in_history_writes_accepted_and_rejected():
    history = [{"query": "q"}]
    info = {"a": {"caption": "a red car"}, "b": {"caption": "a blue bicycle"}}
    record_feedback_in_history(history, info, ["a"], ["b"])
    assert history[-1]["accepted"] == ["a red car"]
    assert history[-1]["rejected"] == ["a blue bicycle"]


def test_record_feedback_in_history_accumulates_without_duplicates():
    history = [{"query": "q"}]
    info = {"a": {"caption": "a red car"}, "b": {"caption": "a blue bicycle"}}
    record_feedback_in_history(history, info, ["a"], [])
    record_feedback_in_history(history, info, ["a", "b"], [])
    assert history[-1]["accepted"].count("a red car") == 1
    assert len(history[-1]["accepted"]) <= 3


def test_record_feedback_in_history_unknown_id_still_recorded():
    history = [{"query": "q"}]
    record_feedback_in_history(history, {}, ["unknown-id"], [])
    assert history[-1]["accepted"] == ["unknown-id"]


def _run_all():
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {test.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()

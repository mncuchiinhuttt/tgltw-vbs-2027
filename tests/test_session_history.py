"""
Unit tests for inference-code/search/session_history.py (per-task session
state: history trimming and the displayed-candidate cache).

Pure logic test - stub dicts only, no LLM / VLM / network access.
Runnable both under pytest and as a plain script:
    python tests/test_session_history.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.conversational_context import (
    describe_candidates,
    format_history,
    record_feedback_in_history,
)
from search.session_history import (
    MAX_HISTORY_TURNS,
    build_candidate_info,
    trim_history,
)


# --- build_candidate_info ---

def test_build_candidate_info_keys_by_id_with_source_and_caption():
    results = [{"id": "p1", "payload": {"source_file": "V001.mp4", "caption": "a red car"}}]
    assert build_candidate_info(results) == {"p1": {"source_file": "V001.mp4", "caption": "a red car"}}


def test_build_candidate_info_truncates_caption():
    results = [{"id": "p1", "payload": {"caption": "x" * 500}}]
    assert len(build_candidate_info(results)["p1"]["caption"]) == 200


def test_build_candidate_info_skips_entries_without_id():
    # /api/temporal-search rows carry no point id - they must not become a
    # None key that later shadows a real candidate.
    results = [{"payload": {"caption": "chain row"}}, {"id": "p1", "payload": {}}]
    info = build_candidate_info(results)
    assert list(info) == ["p1"]
    assert None not in info


def test_build_candidate_info_handles_missing_payload_and_empty_input():
    assert build_candidate_info([]) == {}
    assert build_candidate_info(None) == {}
    assert build_candidate_info([{"id": "p1"}]) == {"p1": {"source_file": None, "caption": ""}}


def test_build_candidate_info_output_feeds_describe_candidates():
    # The whole point of the cache: ids present here must describe as words,
    # never as a raw UUID fallback.
    results = [{"id": "3f2a-uuid", "payload": {"caption": "a man walking a dog"}}]
    info = build_candidate_info(results)
    assert describe_candidates(info, ["3f2a-uuid"]) == ["a man walking a dog"]
    # and an id absent from the cache is exactly the noise case being fixed
    assert describe_candidates({}, ["3f2a-uuid"]) == ["3f2a-uuid"]


def test_two_feedback_rounds_never_leak_raw_ids_into_the_prompt():
    # Regression for the real sequence: /api/feedback returns a NEW result set
    # that replaces what the operator sees, but used to leave
    # last_candidate_info pointing at the previous search's pool. A second
    # round of feedback then described its picks as raw point UUIDs.
    history = [{"query": "một người đàn ông dắt chó"}]

    search_results = [{"id": "s1", "payload": {"caption": "a man cycling"}}]
    record_feedback_in_history(history, build_candidate_info(search_results), [], ["s1"])

    # /api/feedback's own hits now replace the display - the cache must follow.
    feedback_results = [{"id": "f9", "payload": {"caption": "a man walking a dog at night"}}]
    record_feedback_in_history(history, build_candidate_info(feedback_results), [], ["f9"])

    rendered = format_history(history)
    assert "a man walking a dog at night" in rendered
    assert "f9" not in rendered, "raw point id leaked into the CQR prompt"


# --- trim_history ---

def test_trim_history_shorter_than_cap_is_untouched():
    history = [{"query": f"q{i}"} for i in range(3)]
    trim_history(history, max_turns=8)
    assert [t["query"] for t in history] == ["q0", "q1", "q2"]


def test_trim_history_keeps_first_turn_and_most_recent():
    history = [{"query": f"q{i}"} for i in range(10)]
    trim_history(history, max_turns=4)
    # q0 is the KIS-C anchor (oracle's opening description) and must survive;
    # the rest of the budget goes to the newest turns.
    assert [t["query"] for t in history] == ["q0", "q7", "q8", "q9"]


def test_trim_history_mutates_in_place():
    history = [{"query": f"q{i}"} for i in range(10)]
    same_ref = history
    trim_history(history, max_turns=3)
    assert same_ref is history
    assert len(history) == 3


def test_trim_history_exactly_at_cap_is_untouched():
    history = [{"query": f"q{i}"} for i in range(5)]
    trim_history(history, max_turns=5)
    assert len(history) == 5


def test_trim_history_empty_and_nonpositive_cap_do_not_raise():
    history = []
    trim_history(history, max_turns=8)
    assert history == []
    unchanged = [{"query": "q0"}, {"query": "q1"}]
    trim_history(unchanged, max_turns=0)
    assert len(unchanged) == 2


def test_trim_history_cap_of_one_keeps_only_the_anchor():
    history = [{"query": f"q{i}"} for i in range(5)]
    trim_history(history, max_turns=1)
    assert [t["query"] for t in history] == ["q0"]


def test_max_history_turns_is_a_sane_default():
    assert 1 < MAX_HISTORY_TURNS <= 20


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

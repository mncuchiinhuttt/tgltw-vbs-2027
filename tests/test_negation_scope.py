"""
Unit tests for inference-code/search/negation_scope.py (splitting a KIS-C
clarification answer into what it affirms vs what it rules out).

Pure logic test - strings only, no LLM / VLM / network access.
Runnable both under pytest and as a plain script:
    python tests/test_negation_scope.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.negation_scope import split_negation_scope


def _sets(text):
    affirmed, negated = split_negation_scope(text)
    return set(affirmed.split()), set(negated.split())


# --- the affirmative case must be untouched ---

def test_plain_affirmative_answer_negates_nothing():
    affirmed, negated = split_negation_scope("the jacket is blue")
    assert affirmed == "the jacket is blue"
    assert negated == ""


def test_empty_and_none_do_not_raise():
    assert split_negation_scope("") == ("", "")
    assert split_negation_scope(None) == ("", "")


# --- English negation ---

def test_english_negation_scopes_to_its_clause():
    affirmed, negated = _sets("no, not red, the jacket is blue")
    assert "red" in negated
    assert {"jacket", "blue"} <= affirmed
    assert "blue" not in negated


def test_english_cue_negates_only_what_follows_it_in_the_clause():
    affirmed, negated = _sets("the jacket is not red")
    assert {"the", "jacket", "is"} <= affirmed
    assert negated == {"red"}


def test_without_is_treated_as_a_negation_cue():
    _, negated = _sets("a street without cars")
    assert "cars" in negated


# --- Vietnamese negation ---

def test_vietnamese_khong_phai_negates_the_rest_of_the_clause():
    affirmed, negated = _sets("không phải màu đỏ, áo màu xanh")
    assert "đỏ" in negated
    assert {"áo", "xanh"} <= affirmed
    # "phải" is consumed as part of the cue - it is a content word elsewhere
    # ("bên phải") and must never be used to penalise a candidate.
    assert "phải" not in negated


def test_vietnamese_cue_mid_clause_keeps_earlier_words_affirmed():
    affirmed, negated = _sets("áo không có màu đỏ")
    assert "áo" in affirmed
    assert "đỏ" in negated


def test_vietnamese_without_diacritics_is_also_recognised():
    _, negated = _sets("khong phai mau do")
    assert "do" in negated
    assert "phai" not in negated


# --- clause boundaries ---

def test_negation_does_not_leak_past_but():
    affirmed, negated = _sets("not a car but a motorbike")
    assert "car" in negated
    assert "motorbike" in affirmed


def test_negation_does_not_leak_past_nhung():
    affirmed, negated = _sets("không phải xe hơi nhưng là xe máy")
    assert "hơi" in negated
    assert "máy" in affirmed


def test_multiple_clauses_each_scope_independently():
    affirmed, negated = _sets("not red, not green, it is blue")
    assert {"red", "green"} <= negated
    assert "blue" in affirmed


def test_bare_no_negates_nothing_on_its_own():
    affirmed, negated = _sets("no, the jacket is blue")
    assert negated == set()
    assert "blue" in affirmed


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

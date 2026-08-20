"""
Unit tests for inference-code/search/kis_c_scoring.py (KIS-C ambiguity
signals + clarification-answer boost).

Pure logic test - stub dicts only, no model / network / Qdrant access.
Runnable both under pytest and as a plain script:
    python tests/test_kis_c_scoring.py
"""
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.kis_c_scoring import (
    distinct_video_ratio,
    score_margin_ambiguity,
    combine_ambiguity_signals,
    boost_by_clarification_answer,
    MARGIN_WEIGHT,
)


def _cand(cid="p1", video="V001.mp4", score=0.05, text_blob="a red car on a street"):
    return {"id": cid, "rrf_score": score, "payload": {"source_file": video, "text_blob": text_blob}}


# --- distinct_video_ratio ---

def test_distinct_video_ratio_all_same_video():
    candidates = [_cand(cid=f"p{i}", video="V001.mp4") for i in range(5)]
    assert distinct_video_ratio(candidates) == 1 / 5


def test_distinct_video_ratio_all_different():
    candidates = [_cand(cid=f"p{i}", video=f"V{i:03d}.mp4") for i in range(5)]
    assert distinct_video_ratio(candidates) == 1.0


def test_distinct_video_ratio_empty_returns_zero():
    assert distinct_video_ratio([]) == 0.0


# --- score_margin_ambiguity ---

def test_score_margin_ambiguity_runaway_winner_is_low():
    candidates = [_cand(cid="p1", score=1.0), _cand(cid="p2", score=0.1)]
    assert math.isclose(score_margin_ambiguity(candidates), 0.1, abs_tol=1e-9)


def test_score_margin_ambiguity_tied_scores_is_high():
    candidates = [_cand(cid="p1", score=1.0), _cand(cid="p2", score=1.0)]
    assert score_margin_ambiguity(candidates) == 1.0


def test_score_margin_ambiguity_single_candidate_returns_zero():
    assert score_margin_ambiguity([_cand(cid="p1", score=1.0)]) == 0.0


def test_score_margin_ambiguity_zero_scores_returns_zero():
    candidates = [_cand(cid="p1", score=0.0), _cand(cid="p2", score=0.0)]
    assert score_margin_ambiguity(candidates) == 0.0


def test_score_margin_ambiguity_ignores_input_order():
    ordered = [_cand(cid="p1", score=1.0), _cand(cid="p2", score=0.1)]
    reversed_ = [_cand(cid="p2", score=0.1), _cand(cid="p1", score=1.0)]
    assert score_margin_ambiguity(ordered) == score_margin_ambiguity(reversed_)


# --- combine_ambiguity_signals ---

def test_combine_ambiguity_signals_is_within_unit_range():
    assert combine_ambiguity_signals(1.0, 1.0) == 1.0
    assert combine_ambiguity_signals(0.0, 0.0) == 0.0
    assert 0.0 <= combine_ambiguity_signals(2.0, -1.0) <= 1.0


def test_combined_score_tied_distinct_pool_still_triggers():
    candidates = [_cand(cid=f"p{i}", video=f"V{i:03d}.mp4", score=1.0) for i in range(10)]
    distinct_ratio = distinct_video_ratio(candidates)
    margin_ambiguity = score_margin_ambiguity(candidates)
    assert combine_ambiguity_signals(distinct_ratio, margin_ambiguity) >= 0.7


def test_combined_score_clear_winner_does_not_trigger():
    scores = [1.0] + [0.05] * 9
    candidates = [_cand(cid=f"p{i}", video=f"V{i:03d}.mp4", score=s) for i, s in enumerate(scores)]
    distinct_ratio = distinct_video_ratio(candidates)
    margin_ambiguity = score_margin_ambiguity(candidates)
    assert combine_ambiguity_signals(distinct_ratio, margin_ambiguity) < 0.7


def test_combined_score_single_candidate_does_not_trigger():
    candidates = [_cand(cid="p1", video="V001.mp4", score=1.0)]
    distinct_ratio = distinct_video_ratio(candidates)
    margin_ambiguity = score_margin_ambiguity(candidates)
    assert combine_ambiguity_signals(distinct_ratio, margin_ambiguity) < 0.7


def test_margin_weight_is_one_so_distinct_ratio_is_inert():
    # The pool reaching compute_ambiguity_score has already been through
    # diversify_by_scene, so distinct_video_ratio is a near-constant 1.0 and
    # was measured to HALVE the ambiguous/unambiguous separation. It must no
    # longer move the combined score at all.
    assert MARGIN_WEIGHT == 1.0
    scores = {combine_ambiguity_signals(d / 10.0, 0.42) for d in range(11)}
    assert scores == {0.42}


def test_score_margin_ambiguity_equals_top2_over_top1():
    # The property that makes AMBIGUITY_THRESHOLD directly interpretable:
    # a threshold of X means "ask when the runner-up scores >= X x the winner".
    for top1, top2 in ((1.0, 0.5), (0.8, 0.2), (0.25, 0.2), (1.0, 1.0)):
        candidates = [_cand(cid="p1", score=top1), _cand(cid="p2", score=top2)]
        assert math.isclose(score_margin_ambiguity(candidates), top2 / top1, abs_tol=1e-9)


def test_combined_score_uses_the_full_unit_range():
    # The old 0.5/0.5 blend pinned the score into [0.5, 1.0], which made every
    # AMBIGUITY_THRESHOLD at or below 0.5 fire unconditionally - half the dial
    # was dead. A runaway winner must now be able to score below 0.5.
    runaway = [_cand(cid="p1", video="V001.mp4", score=1.0),
               _cand(cid="p2", video="V002.mp4", score=0.05)]
    combined = combine_ambiguity_signals(
        distinct_video_ratio(runaway), score_margin_ambiguity(runaway)
    )
    assert combined < 0.5


# --- boost_by_clarification_answer ---

def test_boost_raises_matching_candidate_above_higher_scored_one():
    # b starts lower-scored than a but fully matches the answer ("car","red"
    # both appear in b's text, neither in a's) - the boost must be enough to
    # overturn a modest initial gap.
    a = _cand(cid="a", video="V001.mp4", score=1.0, text_blob="a blue bicycle in a park")
    b = _cand(cid="b", video="V002.mp4", score=0.6, text_blob="a red car on a street")
    result = boost_by_clarification_answer([a, b], ["a", "b"], "the car was red")
    assert result[0]["id"] == "b"


def test_boost_ignores_candidates_not_in_prior_ids():
    a = _cand(cid="a", video="V001.mp4", score=1.0, text_blob="a blue bicycle")
    b = _cand(cid="b", video="V002.mp4", score=0.5, text_blob="a red car on a street")
    result = boost_by_clarification_answer([a, b], ["a"], "the car was red")
    by_id = {c["id"]: c for c in result}
    assert by_id["b"]["rrf_score"] == 0.5


def test_boost_empty_answer_is_noop():
    a = _cand(cid="a", score=1.0)
    b = _cand(cid="b", score=0.5)
    original = [a, b]
    result = boost_by_clarification_answer(original, ["a", "b"], "")
    assert result is original
    assert [c["rrf_score"] for c in result] == [1.0, 0.5]


def test_boost_empty_prior_ids_is_noop():
    a = _cand(cid="a", score=1.0)
    original = [a]
    result = boost_by_clarification_answer(original, [], "red car")
    assert result is original


def test_boost_no_token_overlap_leaves_scores_unchanged():
    a = _cand(cid="a", score=1.0, text_blob="a blue bicycle in a park")
    result = boost_by_clarification_answer([a], ["a"], "a yellow submarine")
    assert result[0]["rrf_score"] == 1.0


def test_boost_never_lowers_a_score():
    a = _cand(cid="a", score=1.0, text_blob="a red car on a street")
    b = _cand(cid="b", score=0.5, text_blob="a blue bicycle")
    before = {"a": 1.0, "b": 0.5}
    result = boost_by_clarification_answer([a, b], ["a", "b"], "red car")
    for c in result:
        assert c["rrf_score"] >= before[c["id"]]


def test_boost_vietnamese_answer_matches_vietnamese_caption():
    # "áo" and "đỏ" (2-char, diacritic) both appear verbatim in b's caption
    # and neither in a's - guards the MIN_TOKEN_LEN=2 + re.UNICODE decisions.
    a = _cand(cid="a", score=1.0, text_blob="một người đàn ông đi xe đạp")
    b = _cand(cid="b", score=0.6, text_blob="người mặc áo đỏ đi trên phố")
    result = boost_by_clarification_answer([a, b], ["a", "b"], "áo đỏ")
    assert result[0]["id"] == "b"


def test_boost_falls_back_to_caption_when_text_blob_missing():
    a = {"id": "a", "rrf_score": 1.0, "payload": {"source_file": "V001.mp4", "caption": "a blue bicycle"}}
    b = {
        "id": "b", "rrf_score": 0.6,
        "payload": {
            "source_file": "V002.mp4",
            "caption": "a street scene",
            "ocr_text": "STOP sign",
            "detected_objects": [{"label": "red car"}],
        },
    }
    result = boost_by_clarification_answer([a, b], ["a", "b"], "the red car")
    assert result[0]["id"] == "b"


def test_boost_handles_missing_payload_and_missing_scores():
    a = {"id": "a"}
    result = boost_by_clarification_answer([a], ["a"], "red car")
    assert result == [a]


def test_boost_negated_answer_picks_the_affirmed_candidate_english():
    # The exact failure this was written for: flat bag-of-words matching gave
    # BOTH candidates the same overlap (the negated word "red" is still
    # lexically present in the red candidate), so the prior ranking won and the
    # ruled-out candidate stayed on top.
    red = _cand(cid="red", video="V1.mp4", score=0.90,
                text_blob="a man in a red jacket walking a dog in a park")
    blue = _cand(cid="blue", video="V2.mp4", score=0.80,
                 text_blob="a man in a blue jacket walking a dog in a park")
    result = boost_by_clarification_answer(
        [red, blue], ["red", "blue"], "no, not red, the jacket is blue")
    assert result[0]["id"] == "blue"


def test_boost_negated_answer_picks_the_affirmed_candidate_vietnamese():
    do = _cand(cid="do", video="V1.mp4", score=0.90,
               text_blob="một người đàn ông mặc áo đỏ dắt chó trong công viên")
    xanh = _cand(cid="xanh", video="V2.mp4", score=0.80,
                 text_blob="một người đàn ông mặc áo xanh dắt chó trong công viên")
    result = boost_by_clarification_answer(
        [do, xanh], ["do", "xanh"], "không phải màu đỏ, áo màu xanh")
    assert result[0]["id"] == "xanh"


def test_boost_penalises_a_purely_negated_answer():
    a = _cand(cid="a", score=1.0, text_blob="a red car on a street")
    result = boost_by_clarification_answer([a], ["a"], "it is not red")
    assert result[0]["rrf_score"] < 1.0
    assert result[0]["clarification_overlap"] < 0


def test_boost_never_drives_a_score_below_zero():
    a = _cand(cid="a", score=0.01, text_blob="a red car on a street")
    result = boost_by_clarification_answer([a], ["a"], "not red")
    assert result[0]["rrf_score"] >= 0.0


def test_boost_affirmative_answer_is_unaffected_by_the_negation_split():
    # Regression guard: the common case must behave exactly as before.
    a = _cand(cid="a", video="V1.mp4", score=1.0, text_blob="a blue bicycle in a park")
    b = _cand(cid="b", video="V2.mp4", score=0.6, text_blob="a red car on a street")
    result = boost_by_clarification_answer([a, b], ["a", "b"], "the car was red")
    assert result[0]["id"] == "b"
    assert result[0]["clarification_overlap"] > 0


def test_boost_token_on_both_sides_carries_no_signal():
    # "màu" appears in both the negated ("không phải màu đỏ") and the affirmed
    # ("áo màu xanh") half, so it cannot discriminate and must contribute
    # nothing. This candidate shares ONLY "màu" with the answer - if the shared
    # token still counted, it would be boosted here.
    a = _cand(cid="a", score=1.0, text_blob="xe màu tím")
    result = boost_by_clarification_answer([a], ["a"], "không phải màu đỏ, áo màu xanh")
    assert result[0]["rrf_score"] == 1.0
    assert result[0].get("clarification_overlap") is None


def test_boost_stopwords_alone_do_not_boost():
    a = _cand(cid="a", score=1.0, text_blob="a red car on a street")
    result = boost_by_clarification_answer([a], ["a"], "it is in the")
    assert result[0]["rrf_score"] == 1.0


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

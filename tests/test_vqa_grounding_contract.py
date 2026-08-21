"""Contract tests for live Type 2 VQA grounding and fail-closed behavior."""

import os
import sys

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.reranker import (  # noqa: E402
    Reranker,
    _parse_vlm_score,
    parse_grounded_vqa_response,
)


class FakeVLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, image, prompt):
        self.calls.append((image, prompt))
        if not self.responses:
            raise AssertionError("unexpected VLM call")
        return self.responses.pop(0)


def _candidate(candidate_id, source_file, **payload):
    return {
        "id": candidate_id,
        "rrf_score": 0.4,
        "payload": {"source_file": source_file, "timestamp": 0.0, **payload},
    }


def test_grounded_vqa_parser_requires_found_answer_and_bounded_confidence():
    valid = parse_grounded_vqa_response(
        '{"found": true, "answer": "a red car", "confidence": 0.8}'
    )
    assert valid["valid"] is True
    assert valid["answer"] == "a red car"
    assert valid["confidence"] == 0.8

    for raw in (
        "not json",
        '{"answer": "a car", "confidence": 0.8}',
        '{"found": true, "answer": "UNKNOWN", "confidence": 0.8}',
        '{"found": true, "answer": "a car", "confidence": 1.1}',
        '{"found": false, "answer": "UNKNOWN", "confidence": 0.0}',
    ):
        result = parse_grounded_vqa_response(raw)
        assert result["valid"] is False
        assert result["answer"] == "UNKNOWN"
        assert result["confidence"] == 0.0


def test_vlm_score_parser_rejects_out_of_range_values():
    assert _parse_vlm_score("Score: 0.75") == 0.75
    assert _parse_vlm_score("Score: 2.0") is None
    assert _parse_vlm_score("Score: -0.1") is None


def test_missing_frame_fails_closed_without_calling_vlm(tmp_path):
    vlm = FakeVLM([])
    reranker = Reranker(vlm)
    result = reranker.rerank_type2_vqa(
        "Is there a person?", ["person"], [_candidate("missing", "missing.mp4")], str(tmp_path)
    )[0]

    assert vlm.calls == []
    assert result["vqa_evidence_available"] is False
    assert result["vqa_answer_valid"] is False
    assert result["vqa_answer"] == "UNKNOWN"
    assert result["vqa_score"] == 0.0
    assert result["vqa_evidence_reason"] == "frame_unavailable"


def test_answers_stay_bound_to_their_candidate_frame(tmp_path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (32, 24), color="red").save(first_path)
    Image.new("RGB", (32, 24), color="blue").save(second_path)
    vlm = FakeVLM([
        '{"found": true, "answer": "red object", "confidence": 0.9}',
        '{"found": true, "answer": "blue object", "confidence": 0.8}',
    ])
    reranker = Reranker(vlm)
    candidates = [
        _candidate("first", "first.mp4", keyframe_path=str(first_path), frame_idx=11),
        _candidate("second", "second.mp4", keyframe_path=str(second_path), frame_idx=22),
    ]

    result = reranker.rerank_type2_vqa("What is visible?", [], candidates, str(tmp_path))
    by_id = {hit["id"]: hit for hit in result}
    assert by_id["first"]["vqa_answer"] == "red object"
    assert by_id["first"]["vqa_frame_idx"] == 11
    assert by_id["first"]["vqa_candidate_id"] == "first"
    assert by_id["second"]["vqa_answer"] == "blue object"
    assert by_id["second"]["vqa_frame_idx"] == 22


def test_invalid_vqa_response_gets_zero_not_neutral_score(tmp_path):
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (20, 20), color="white").save(frame_path)
    vlm = FakeVLM(['{"answer": "YES", "confidence": 0.9}'])
    reranker = Reranker(vlm)
    result = reranker.rerank_type2_vqa(
        "Is it present?", [], [_candidate("one", "frame.mp4", keyframe_path=str(frame_path))], str(tmp_path)
    )[0]

    assert result["vqa_evidence_available"] is True
    assert result["vqa_answer_valid"] is False
    assert result["vqa_score"] == 0.0
    assert result["vqa_answer"] == "UNKNOWN"


def test_crop_bounding_box_supports_normalized_coordinates():
    image = Image.new("RGB", (100, 80), color="white")
    crop = Reranker(FakeVLM([])).crop_bounding_box(image, [0.1, 0.25, 0.5, 0.75])
    assert crop.size == (40, 40)


def test_verification_question_parser_does_not_fallback_to_raw_query():
    vlm = FakeVLM(['{"questions": ["only one"]}'])
    reranker = Reranker(vlm)
    assert reranker.generate_verification_questions("a person", n=2) == []

"""
Unit tests for preprocessing.audio.asr_segment_filter.filter_asr_segments.

Pure logic test - stub dicts only, no ASR model / torch / network access.
Runnable both under pytest and as a plain script:
    python tests/test_asr_segment_filter.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.audio.asr_segment_filter import filter_asr_segments


def _seg(text="hello world", avg_logprob=-0.3, no_speech_prob=0.1, compression_ratio=1.5):
    return {
        "text": text,
        "start": 0.0,
        "end": 1.0,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
        "words": [],
    }


def test_keeps_good_segment():
    result = filter_asr_segments([_seg()])
    assert len(result) == 1


def test_drops_empty_text():
    result = filter_asr_segments([_seg(text=""), _seg(text=" ")])
    assert result == []


def test_drops_high_no_speech_prob():
    result = filter_asr_segments([_seg(no_speech_prob=0.95)])
    assert result == []


def test_drops_low_avg_logprob():
    result = filter_asr_segments([_seg(avg_logprob=-2.5)])
    assert result == []


def test_drops_high_compression_ratio():
    result = filter_asr_segments([_seg(compression_ratio=5.0)])
    assert result == []


def test_keeps_segment_with_missing_confidence_keys():
    minimal = {"text": "hello world", "start": 0.0, "end": 1.0}
    result = filter_asr_segments([minimal])
    assert len(result) == 1


def test_empty_input_returns_empty_list():
    assert filter_asr_segments([]) == []


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

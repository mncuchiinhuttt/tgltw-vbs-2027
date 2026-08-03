from typing import List, Dict, Any

from preprocessing.config import ASR_MIN_AVG_LOGPROB, ASR_MAX_NO_SPEECH_PROB, ASR_MAX_COMPRESSION_RATIO


def filter_asr_segments(
    segments: List[Dict[str, Any]],
    min_avg_logprob: float = ASR_MIN_AVG_LOGPROB,
    max_no_speech_prob: float = ASR_MAX_NO_SPEECH_PROB,
    max_compression_ratio: float = ASR_MAX_COMPRESSION_RATIO,
    min_chars: int = 2,
) -> List[Dict[str, Any]]:
    """
    Drop ASR segments that are empty, silence, or Whisper hallucinations
    (repeated/garbage text over non-speech audio) before they reach
    embedding + indexing. Confidence keys missing from a segment are
    treated as passing (permissive), so a future ASR backend that doesn't
    report these never silently drops everything.
    """
    kept = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        no_speech_prob = seg.get("no_speech_prob")
        if no_speech_prob is not None and no_speech_prob > max_no_speech_prob:
            continue
        avg_logprob = seg.get("avg_logprob")
        if avg_logprob is not None and avg_logprob < min_avg_logprob:
            continue
        compression_ratio = seg.get("compression_ratio")
        if compression_ratio is not None and compression_ratio > max_compression_ratio:
            continue
        kept.append(seg)
    return kept

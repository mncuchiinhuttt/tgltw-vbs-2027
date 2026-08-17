"""
Pure-Python KIS-C (VBS conversational Known-Item-Search) scoring helpers.

Stdlib-only (`re` plus the sibling `negation_scope`) - no numpy, no
qdrant_client, no config import, no LLM/VLM call. Factored out of
HybridSearcher so it stays unit-testable without a QdrantClient, same
rationale as `preprocessing/audio/asr_segment_filter.py`. Imported by
`hybrid_search.py` (ambiguity scoring) and `webapp/backend/main.py`
(clarification-answer boost wiring).
"""
import re

from search.negation_scope import split_negation_scope

# Weight of the score-margin signal in the combined ambiguity score (see
# combine_ambiguity_signals). 1.0 = margin only, distinct-video ratio ignored.
# Why: compute_ambiguity_score runs AFTER diversify_by_scene (one candidate per
# source_file+scene_id), so its top-10 is almost always 10 distinct videos and
# distinct_video_ratio sat at a constant 1.0 - a fixed offset, not information.
# Separation between an ambiguous pool and the worst unambiguous one, replaying
# the real pipeline: margin only +0.275; old 0.5/0.5 blend +0.138 (the constant
# term halved it); top-1 share +0.037; per-video mass +0.037; entropy +0.004
# (saturates - RRF scores are near-flat). Every replacement measured WORSE than
# dropping the term; the blend is kept so it can be weighed back in later.
MARGIN_WEIGHT = 1.0

# Max fraction of the pool's top score boost_by_clarification_answer can add
# to a fully-matching candidate (or subtract from a fully-contradicted one).
CLARIFICATION_BOOST_WEIGHT = 0.5

# Vietnamese content words are often 2 chars ("áo", "xe") - never raise to 3.
MIN_TOKEN_LEN = 2

# Function words only (EN + VI). Do NOT add content words like
# "người"/"xe"/"màu" - that would suppress genuine clarification-answer signal.
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "is", "are",
    "was", "were", "it", "its", "this", "that", "with", "for", "co",
    "la", "va", "cua", "duoc", "thi", "ma", "cac", "nhung", "mot", "cai",
    "dang", "o", "voi", "cho", "roi", "thay",
    "có", "là", "và", "của", "được", "thì", "mà", "các", "những", "một",
    "cái", "đang", "ở", "với", "cho", "rồi", "thấy",
})


def distinct_video_ratio(candidates: list, top_n: int = 10) -> float:
    """
    Ratio of distinct source videos among the top `top_n` candidates - the
    original compute_ambiguity_score logic, moved verbatim. 0.0 (empty slice)
    to 1.0 (every candidate a different video).
    """
    top = candidates[:top_n] if candidates else []
    if not top:
        return 0.0
    distinct_videos = {(c.get("payload") or {}).get("source_file") for c in top}
    return len(distinct_videos) / len(top)


def score_margin_ambiguity(candidates: list, top_n: int = 10, score_key: str = "rrf_score") -> float:
    """
    Normalized top-1 vs top-2 score margin among the top `top_n` candidates:
    a big lead for the top candidate means low ambiguity, a near-tie means
    high ambiguity. Complements distinct_video_ratio, which only counts
    videos and is blind to how confidently the fusion actually ranked them
    (a 10-distinct-video pool with a runaway winner and one with 10 tied
    scores both score 1.0 under the count-only metric).

    Meaningful at the real call site because HybridSearcher.compute_ambiguity_score
    runs *after* temporal_coherence_boost, which spreads out the score
    distribution; on raw RRF scores alone the top-1/top-2 gap is near-zero.
    """
    top = candidates[:top_n] if candidates else []
    scores = sorted((float(c.get(score_key, 0.0)) for c in top), reverse=True)
    if len(scores) < 2 or scores[0] <= 0:
        return 0.0
    top1, top2 = scores[0], scores[1]
    relative_margin = max(0.0, min(1.0, (top1 - top2) / top1))
    return 1.0 - relative_margin


def combine_ambiguity_signals(distinct_ratio: float, margin_ambiguity: float, margin_weight: float = MARGIN_WEIGHT) -> float:
    """
    Weighted average of the two ambiguity signals into a single float on the
    0-1 scale the AMBIGUITY_THRESHOLD env knob is compared against (weighted
    average, not OR-of-thresholds, so callers keep a single interpretable
    number).

    At the default MARGIN_WEIGHT of 1.0 this returns `margin_ambiguity`
    unchanged and ignores `distinct_ratio` (see MARGIN_WEIGHT). That makes the
    number directly interpretable, since score_margin_ambiguity simplifies to
    top2/top1: AMBIGUITY_THRESHOLD = X asks exactly when the runner-up scores
    at least X times the winner. The whole 0-1 range is usable, unlike the old
    0.5/0.5 blend where distinct_ratio's constant 1.0 pinned the score into
    [0.5, 1.0] and made every threshold at or below 0.5 fire unconditionally.
    """
    combined = (1 - margin_weight) * distinct_ratio + margin_weight * margin_ambiguity
    return max(0.0, min(1.0, combined))


def candidate_match_text(payload: dict) -> str:
    """
    Best-effort text blob for a candidate, lowercased. Prefers `text_blob`
    (already concatenates caption + scene narrative + OCR + object labels +
    actions + speech at index time - DRY), falling back to assembling the
    individual fields for candidates indexed before `text_blob` existed.
    """
    payload = payload or {}
    text_blob = payload.get("text_blob")
    if text_blob:
        return str(text_blob).lower()

    parts = [
        str(payload.get("caption") or ""),
        str(payload.get("scene_narrative") or ""),
        str(payload.get("ocr_text") or ""),
        str(payload.get("actions") or ""),
    ]
    for obj in payload.get("detected_objects") or []:
        if isinstance(obj, dict) and obj.get("label"):
            parts.append(str(obj["label"]))
    return " ".join(p for p in parts if p).lower()


def tokenize_answer(text: str) -> set:
    """
    Lowercase word tokens from operator-supplied free text, dropping
    stopwords and tokens shorter than MIN_TOKEN_LEN. `\\w` + re.UNICODE
    already covers Vietnamese diacritics.
    """
    if not text:
        return set()
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    return {t for t in tokens if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS}


def boost_by_clarification_answer(
    candidates: list,
    prior_candidate_ids: list,
    answer_text: str,
    boost_weight: float = CLARIFICATION_BOOST_WEIGHT,
    score_key: str = "rrf_score",
) -> list:
    """
    Clarification-answer -> direct re-rank (Sekulic et al., arXiv:2008.03717,
    "Analysing the Effect of Clarifying Questions on Document Ranking in
    Conversational Search"; +18% recall / +12% nDCG@3 in that paper's setting
    - indicative, not a guarantee here). Additively boosts, on the CURRENT
    turn's normal RRF-fused pool, whichever of the candidates the clarifying
    question was based on (`prior_candidate_ids`) now overlap with the
    operator's answer text. This is deliberately NOT a fast path that skips
    search - the full per-turn pipeline still runs; this only sharpens its
    final ranking.

    `negation_scope.split_negation_scope` splits the answer into what it
    affirms and what it rules out, and a candidate scores on (affirmed_overlap
    - negated_overlap): "không phải màu đỏ, áo màu xanh" must PENALISE the red
    candidate, not boost it for containing "đỏ". Purely affirmative answers
    yield an empty negated set and behave exactly as before; tokens on both
    sides carry no discriminating signal and are dropped from both.
    `clarification_overlap` is therefore signed, and scores floor at 0.0.

    True no-op (returns the exact same `candidates` object, untouched) when
    there is no answer text, no prior ids, or no candidates. When tokens and
    prior ids exist but nothing actually overlaps, a new list is returned
    (same order/content, since the input is already score-sorted) rather
    than the original object - total/never-raises either way, since KIS-C
    runs on a live 7-minute competition clock.
    """
    if not candidates:
        return candidates
    affirmed_text, negated_text = split_negation_scope(answer_text)
    affirmed = tokenize_answer(affirmed_text)
    negated = tokenize_answer(negated_text)
    shared = affirmed & negated
    affirmed -= shared
    negated -= shared
    if not affirmed and not negated:
        return candidates
    prior = set(prior_candidate_ids or ())
    if not prior:
        return candidates

    scores = [float(c.get(score_key, 0.0)) for c in candidates]
    top_score = max(scores) if scores and max(scores) > 0 else 1.0

    for c in candidates:
        if c.get("id") not in prior:
            continue
        candidate_tokens = tokenize_answer(candidate_match_text(c.get("payload")))
        if not candidate_tokens:
            continue
        hit = len(affirmed & candidate_tokens) / len(affirmed) if affirmed else 0.0
        miss = len(negated & candidate_tokens) / len(negated) if negated else 0.0
        delta = hit - miss
        if delta == 0:
            continue
        c[score_key] = max(0.0, c.get(score_key, 0.0) + boost_weight * delta * top_score)
        c["clarification_overlap"] = round(delta, 3)

    return sorted(candidates, key=lambda c: c.get(score_key, 0.0), reverse=True)

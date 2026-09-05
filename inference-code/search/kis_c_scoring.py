"""
Pure-Python KIS-C (VBS conversational Known-Item-Search) scoring helpers.

Everything here is stdlib-only (just `re`) - no numpy, no qdrant_client, no
config import, no LLM/VLM call. Factored out of HybridSearcher so it stays
unit-testable without a QdrantClient, same rationale as
`webapp/backend/main.py`'s `_check_avs_duplicate` and
`preprocessing/audio/asr_segment_filter.py`. Imported by
`hybrid_search.py` (ambiguity scoring) and `webapp/backend/main.py`
(clarification-answer boost wiring).
"""
import re

# Weight of the score-margin signal inside the combined ambiguity score
# (see combine_ambiguity_signals). 0.5 = equal weight with the pre-existing
# distinct-video ratio.
MARGIN_WEIGHT = 0.5

# Max fraction of the pool's top score a fully-matching candidate can gain
# from boost_by_clarification_answer.
CLARIFICATION_BOOST_WEIGHT = 0.5

# Vietnamese content words are frequently 2 characters ("áo", "xe") - do not
# raise this to 3, it would silently drop real signal.
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
    same 0-1 scale the existing AMBIGUITY_THRESHOLD env knob is tuned
    against (weighted average, not OR-of-thresholds, so callers keep a single
    interpretable number).
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


def extract_phrases(text: str) -> set:
    """
    Extracts bi-gram and tri-gram phrases to capture compound visual concepts
    (e.g., 'traditional wedding', 'peacock theme', 'áo đỏ', 'xe máy').
    """
    if not text:
        return set()
    words = [w for w in re.findall(r"\w+", text.lower(), re.UNICODE) if w not in STOPWORDS and len(w) >= MIN_TOKEN_LEN]
    phrases = set()
    for i in range(len(words) - 1):
        phrases.add(f"{words[i]} {words[i+1]}")
    for i in range(len(words) - 2):
        phrases.add(f"{words[i]} {words[i+1]} {words[i+2]}")
    return phrases

def compute_semantic_overlap(answer_text: str, candidate_text: str) -> float:
    """
    Computes hybrid unigram + compound phrase overlap between operator answer
    and candidate text payload.
    """
    if not answer_text or not candidate_text:
        return 0.0
    
    cand_lower = candidate_text.lower()
    answer_tokens = tokenize_answer(answer_text)
    if not answer_tokens:
        return 0.0

    cand_tokens = tokenize_answer(cand_lower)
    unigram_overlap = len(answer_tokens & cand_tokens) / len(answer_tokens) if answer_tokens else 0.0

    # Phrase / N-gram bonus when consecutive multi-word phrases match
    answer_phrases = extract_phrases(answer_text)
    phrase_bonus = 0.0
    if answer_phrases:
        matched_phrases = sum(1 for p in answer_phrases if p in cand_lower)
        phrase_bonus = 0.25 * (matched_phrases / len(answer_phrases))

    # Base unigram overlap plus phrase bonus, capped at 1.0
    combined = min(1.0, unigram_overlap + phrase_bonus)
    return round(combined, 4)


def boost_by_clarification_answer(
    candidates: list,
    prior_candidate_ids: list,
    answer_text: str,
    boost_weight: float = CLARIFICATION_BOOST_WEIGHT,
    score_key: str = "rrf_score",
) -> list:
    """
    Clarification-answer -> direct re-rank with compound phrase & semantic overlap.
    """
    if not candidates:
        return candidates
    if not answer_text or not answer_text.strip():
        return candidates
    prior = set(prior_candidate_ids or ())
    scores = [float(c.get(score_key, 0.0)) for c in candidates]
    top_score = max(scores) if scores and max(scores) > 0 else 1.0

    for c in candidates:
        cand_text = candidate_match_text(c.get("payload"))
        if not cand_text:
            continue
        overlap = compute_semantic_overlap(answer_text, cand_text)
        if overlap <= 0:
            continue
        in_prior = not prior or c.get("id") in prior
        weight = boost_weight if in_prior else (boost_weight * 0.85)
        c[score_key] = c.get(score_key, 0.0) + weight * overlap * top_score
        c["clarification_overlap"] = round(overlap, 3)
    return sorted(candidates, key=lambda c: c.get(score_key, 0.0), reverse=True)


def apply_conversational_negative_filter(
    candidates: list,
    rejected_descriptions: list,
    penalty_weight: float = 0.35,
    score_key: str = "rrf_score",
) -> list:
    """
    Applies a calibrated penalty to candidates that match concepts explicitly rejected
    by the operator during conversational exploration.
    """
    if not candidates or not rejected_descriptions:
        return candidates

    rejected_text = " ".join(str(d) for d in rejected_descriptions if d).lower()
    rejected_tokens = tokenize_answer(rejected_text)
    if not rejected_tokens:
        return candidates

    for c in candidates:
        cand_text = candidate_match_text(c.get("payload"))
        if not cand_text:
            continue
        cand_tokens = tokenize_answer(cand_text)
        overlap = len(rejected_tokens & cand_tokens) / len(rejected_tokens)
        if overlap > 0.3:
            current_score = float(c.get(score_key, 0.0))
            penalty = penalty_weight * overlap * current_score
            c[score_key] = max(0.0, current_score - penalty)
            c["negative_feedback_penalty"] = round(penalty, 4)

    return sorted(candidates, key=lambda c: c.get(score_key, 0.0), reverse=True)

"""
Subset selection over a scene's candidate frame embeddings.

Two different questions are asked of the same candidate pool, and they want
different objectives:

* Which frames go into the retrieval index?  A frame that was never indexed
  can never be retrieved, so the objective is *coverage*: keep adding frames
  until every candidate lies within TAU of something that was kept.  That is
  greedy k-center, whose 2-approximation guarantee (Gonzalez 1985,
  doi:10.1016/0304-3975(85)90224-5) is the same
  construction Core-Set active learning uses (Sener & Savarese, ICLR 2018,
  arXiv:1708.00489).  It also replaces a pair of hand-tuned variance
  thresholds whose scale was tied to one CLIP checkpoint's un-normalized
  output - TAU is a cosine distance and means the same thing for any encoder.

* Which of those frames are worth the expensive per-frame passes (VLM, SAM3,
  OCR)?  Here the budget is fixed and small, so the objective is to represent
  the scene as well as possible within it.  Farthest-point sampling maximises
  *dispersion*, which systematically favours the outliers of the pool -
  motion blur, dissolves, near-black frames - because those are what sits
  furthest from everything else.  Facility location maximises *coverage* of
  the pool instead, which is what actually gets asked at query time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def insert_official_candidate(
    candidates: List[Dict[str, Any]], official_candidate: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Place a shot's official keyframe into the decoded candidate list at its
    chronological position and report where it landed.

    Position matters because selection returns candidate indices in list
    order: a shot's frames must stay in time order for the H-EAGLE shot
    payload and for anything else reading the timeline.  The returned index is
    what the caller passes as `forced_indices` so the official keyframe
    survives selection - its identifier is what the competition scores, so it
    has to be indexed even when its content is redundant.
    """
    timestamp = official_candidate["timestamp"]
    position = next(
        (index for index, frame in enumerate(candidates) if frame["timestamp"] > timestamp),
        len(candidates),
    )
    return candidates[:position] + [official_candidate] + candidates[position:], position


def normalize_rows(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Stack embeddings as unit rows so dot products are cosine similarities."""
    matrix = np.stack([np.asarray(vector, dtype=np.float32).ravel() for vector in vectors])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def similarity_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    matrix = normalize_rows(vectors)
    return np.clip(matrix @ matrix.T, -1.0, 1.0)


def _seed_index(similarities: np.ndarray, forced: Sequence[int]) -> int:
    """Start from the most central frame - the best single representative."""
    if len(forced):
        return int(forced[0])
    return int(np.argmax(similarities.sum(axis=1)))


def select_by_coverage(
    vectors: Sequence[np.ndarray],
    tau: float,
    max_budget: int,
    forced_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Greedy k-center: keep adding the worst-covered candidate until every
    candidate is within `tau` cosine distance of a selected one.

    Returns sorted candidate indices.  `forced_indices` are always kept (the
    official keyframe of a shot must stay in the index whatever the geometry
    says, because its identifier is what the competition scores against).
    """
    count = len(vectors)
    if count == 0:
        return []
    forced = sorted({index for index in (forced_indices or []) if 0 <= index < count})
    # A forced frame is kept even if that pushes past the budget: dropping the
    # official keyframe would lose the identifier the competition scores on.
    budget = max(1, min(max_budget, count), len(forced))

    similarities = similarity_matrix(vectors)
    selected = list(forced) or [_seed_index(similarities, forced)]
    # distance of every candidate to its nearest selected frame
    covered = 1.0 - similarities[selected].max(axis=0)

    while len(selected) < budget:
        worst = int(np.argmax(covered))
        if covered[worst] <= tau:
            break
        selected.append(worst)
        covered = np.minimum(covered, 1.0 - similarities[worst])

    return sorted(set(selected))


def select_by_facility_location(
    vectors: Sequence[np.ndarray],
    budget: int,
    quality: Optional[Sequence[float]] = None,
    quality_weight: float = 0.0,
    forced_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Greedy maximisation of sum_i max_{s in S} sim(i, s), i.e. how well the
    selected subset represents the whole pool.  Monotone submodular, so the
    greedy solution is within (1 - 1/e) of optimal (Nemhauser, Wolsey &
    Fisher, Mathematical Programming 14:265-294, 1978,
    doi:10.1007/BF01588971).

    `quality` (Laplacian sharpness, already normalised to [0, 1]) breaks ties
    toward frames that are actually readable, at `quality_weight` strength.
    """
    count = len(vectors)
    if count == 0:
        return []
    forced_count = len({index for index in (forced_indices or []) if 0 <= index < count})
    budget = max(1, min(budget, count), forced_count)
    if count <= budget:
        return list(range(count))

    similarities = similarity_matrix(vectors)
    bonus = np.zeros(count, dtype=np.float32)
    if quality is not None and quality_weight > 0:
        scores = np.asarray([0.0 if value is None else float(value) for value in quality], dtype=np.float32)
        if len(scores) == count:
            bonus = quality_weight * count * np.clip(scores, 0.0, 1.0)

    forced = sorted({index for index in (forced_indices or []) if 0 <= index < count})
    selected = list(forced) or [int(np.argmax(similarities.sum(axis=1) + bonus))]
    best_similarity = similarities[selected].max(axis=0)

    while len(selected) < budget:
        # Marginal gain of adding each remaining candidate.
        gains = np.maximum(similarities, best_similarity).sum(axis=1) + bonus
        gains[selected] = -np.inf
        chosen = int(np.argmax(gains))
        if not np.isfinite(gains[chosen]):
            break
        selected.append(chosen)
        best_similarity = np.maximum(best_similarity, similarities[chosen])

    return sorted(set(selected))

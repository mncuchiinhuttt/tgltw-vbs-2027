"""Index-coverage and VLM-tier subset selection."""

import numpy as np

from preprocessing.video.keyframe_selection import (
    insert_official_candidate,
    select_by_coverage,
    select_by_facility_location,
)


def clustered_pool(seed: int = 0):
    """Three tight clusters (0-4, 5-9, 10-14) plus one far outlier (15).

    The outlier stands in for what farthest-point sampling used to reward: a
    dissolve or motion-blurred frame, which sits further from everything else
    than any real content does.
    """
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(3):
        centre = rng.normal(size=32)
        centre /= np.linalg.norm(centre)
        for _ in range(5):
            member = centre + 0.02 * rng.normal(size=32)
            vectors.append(member / np.linalg.norm(member))
    outlier = rng.normal(size=32)
    vectors.append(outlier / np.linalg.norm(outlier))
    return vectors


def cluster_of(index: int) -> int:
    return index // 5


def test_coverage_keeps_one_frame_per_distinct_cluster():
    selected = select_by_coverage(clustered_pool(), tau=0.12, max_budget=12)

    assert {cluster_of(index) for index in selected} == {0, 1, 2, 3}
    assert len(selected) == 4


def test_coverage_prunes_a_redundant_pool_far_below_its_budget():
    # Five near-identical frames need one representative, not five points.
    selected = select_by_coverage(clustered_pool()[:5], tau=0.12, max_budget=12)

    assert len(selected) == 1


def test_coverage_widens_with_tau():
    pool = clustered_pool()

    assert len(select_by_coverage(pool, tau=0.99, max_budget=12)) < len(
        select_by_coverage(pool, tau=0.001, max_budget=12)
    )


def test_coverage_respects_the_budget_ceiling():
    assert len(select_by_coverage(clustered_pool(), tau=0.0, max_budget=3)) == 3


def test_coverage_always_keeps_forced_frames():
    # The official keyframe carries the identifier the competition scores on,
    # so it stays indexed however redundant its content is.
    selected = select_by_coverage(clustered_pool(), tau=0.12, max_budget=2, forced_indices=[7])

    assert 7 in selected


def test_facility_location_avoids_the_outlier_that_dispersion_would_pick():
    selected = select_by_facility_location(clustered_pool(), budget=3)

    assert 15 not in selected
    assert {cluster_of(index) for index in selected} == {0, 1, 2}


def test_facility_location_respects_budget_and_forced_frames():
    selected = select_by_facility_location(clustered_pool(), budget=3, forced_indices=[15])

    assert 15 in selected
    assert len(selected) == 3


def test_selection_returns_sorted_indices():
    pool = clustered_pool()

    assert select_by_coverage(pool, tau=0.12, max_budget=12) == sorted(
        select_by_coverage(pool, tau=0.12, max_budget=12)
    )
    assert select_by_facility_location(pool, budget=4) == sorted(
        select_by_facility_location(pool, budget=4)
    )


def test_quality_bonus_breaks_ties_toward_sharper_frames():
    # Two identical frames: the sharper one should win the single slot.
    identical = [np.array([1.0, 0.0], dtype=np.float32)] * 2

    selected = select_by_facility_location(
        identical, budget=1, quality=[0.0, 1.0], quality_weight=0.5
    )

    assert selected == [1]


def test_handles_degenerate_pools():
    assert select_by_coverage([], tau=0.1, max_budget=5) == []
    assert select_by_facility_location([], budget=5) == []
    assert select_by_coverage([np.ones(4)], tau=0.1, max_budget=5) == [0]
    assert select_by_facility_location([np.ones(4)], budget=5) == [0]


def test_zero_vectors_do_not_produce_nan_selections():
    pool = [np.zeros(8), np.ones(8), np.zeros(8)]

    assert select_by_coverage(pool, tau=0.12, max_budget=3)
    assert select_by_facility_location(pool, budget=2)


# --- anchoring the official keyframe ---------------------------------------

def decoded(*timestamps):
    return [{"timestamp": float(t), "frame_idx": int(t * 8)} for t in timestamps]


def official(timestamp):
    return {"timestamp": float(timestamp), "frame_idx": 31, "asset_source": "v3c_keyframe"}


def test_official_keyframe_lands_in_chronological_order():
    candidates, position = insert_official_candidate(decoded(0.0, 1.0, 2.0, 3.0), official(1.5))

    assert [c["timestamp"] for c in candidates] == [0.0, 1.0, 1.5, 2.0, 3.0]
    assert position == 2
    assert candidates[position]["asset_source"] == "v3c_keyframe"


def test_official_keyframe_before_every_decoded_frame():
    candidates, position = insert_official_candidate(decoded(1.0, 2.0), official(0.5))

    assert position == 0
    assert [c["timestamp"] for c in candidates] == [0.5, 1.0, 2.0]


def test_official_keyframe_after_every_decoded_frame():
    candidates, position = insert_official_candidate(decoded(1.0, 2.0), official(9.0))

    assert position == 2
    assert [c["timestamp"] for c in candidates] == [1.0, 2.0, 9.0]


def test_official_keyframe_survives_a_failed_decode():
    # If decoding produced nothing, the official keyframe is still the one
    # frame that must reach the index - its identifier is what gets scored.
    candidates, position = insert_official_candidate([], official(1.5))

    assert position == 0
    assert len(candidates) == 1


def test_insertion_does_not_mutate_the_caller_list():
    original = decoded(0.0, 1.0)

    insert_official_candidate(original, official(0.5))

    assert len(original) == 2


def test_the_reported_position_is_what_selection_must_force():
    candidates, position = insert_official_candidate(decoded(0.0, 1.0, 2.0), official(1.5))
    vectors = [np.array([float(c["timestamp"]), 1.0]) for c in candidates]

    # Budget of one, so only a forced index can survive.
    assert select_by_coverage(vectors, tau=0.5, max_budget=1, forced_indices=[position]) == [position]

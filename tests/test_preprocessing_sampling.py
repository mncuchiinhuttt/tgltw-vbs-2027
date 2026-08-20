"""Keyframe quality signal and the index/VLM budget split."""

import numpy as np

from preprocessing.video.scene_detector import (
    compute_laplacian_sharpness,
    select_diverse_keyframes,
    select_index_and_vlm_keyframes,
    subsample_candidates,
)


class FakeEmbedder:
    def embed_image(self, image):
        return np.array([float(np.asarray(image).mean()), 1.0], dtype=np.float32)


def varied_frames(count=16):
    """Frames whose embeddings spread out, so coverage has work to do."""
    return [
        {
            "frame_img": np.full((16, 16, 3), (index * 16) % 256, dtype=np.uint8),
            "timestamp": float(index),
            "frame_idx": index,
        }
        for index in range(count)
    ]


def test_laplacian_sharpness_distinguishes_edges_from_flat_frame():
    flat = np.zeros((32, 32, 3), dtype=np.uint8)
    edges = np.indices((32, 32)).sum(axis=0).astype(np.uint8) * 255
    edges = np.repeat(edges[:, :, None], 3, axis=2)

    assert compute_laplacian_sharpness(flat) == 0.0
    assert compute_laplacian_sharpness(edges) > 0.0


def test_selection_stays_within_budget_and_records_quality():
    frames = [
        {"frame_img": np.full((16, 16, 3), value, dtype=np.uint8), "timestamp": float(i), "frame_idx": i}
        for i, value in enumerate((0, 64, 128, 255))
    ]

    selected = select_diverse_keyframes(frames, FakeEmbedder(), budget=2)

    assert len(selected) == 2
    assert all("sharpness" in frame for frame in selected)
    assert all(0.0 <= frame["sharpness"] <= 1.0 for frame in selected)


def test_more_frames_are_indexed_than_are_sent_to_the_vlm():
    # The whole point of separating the budgets: a frame that was never
    # indexed can never be retrieved, but describing every indexed frame with
    # a VLM would multiply the expensive passes along with the index.
    index_frames, vlm_frames = select_index_and_vlm_keyframes(
        varied_frames(), FakeEmbedder(), vlm_budget=2, index_max_budget=12, coverage_tau=0.001
    )

    assert len(index_frames) > len(vlm_frames)
    assert len(vlm_frames) == 2


def test_the_vlm_tier_is_a_subset_of_the_indexed_tier():
    index_frames, vlm_frames = select_index_and_vlm_keyframes(
        varied_frames(), FakeEmbedder(), vlm_budget=3, index_max_budget=12, coverage_tau=0.001
    )

    indexed_timestamps = {frame["timestamp"] for frame in index_frames}
    assert {frame["timestamp"] for frame in vlm_frames} <= indexed_timestamps


def test_both_tiers_stay_in_chronological_order():
    index_frames, vlm_frames = select_index_and_vlm_keyframes(
        varied_frames(), FakeEmbedder(), vlm_budget=3, index_max_budget=12, coverage_tau=0.001
    )

    assert [f["timestamp"] for f in index_frames] == sorted(f["timestamp"] for f in index_frames)
    assert [f["timestamp"] for f in vlm_frames] == sorted(f["timestamp"] for f in vlm_frames)


def test_the_official_keyframe_survives_both_selection_passes():
    # Its identifier is what the competition scores against, so it has to stay
    # indexed and described however redundant its content is.
    frames = varied_frames()
    official_position = 7
    frames[official_position]["asset_source"] = "v3c_keyframe"

    index_frames, vlm_frames = select_index_and_vlm_keyframes(
        frames, FakeEmbedder(), vlm_budget=2, index_max_budget=4, coverage_tau=0.5,
        forced_indices=[official_position],
    )

    assert any(f.get("asset_source") == "v3c_keyframe" for f in index_frames)
    assert any(f.get("asset_source") == "v3c_keyframe" for f in vlm_frames)


def test_every_candidate_is_embedded_exactly_once():
    frames = varied_frames(8)

    class CountingEmbedder(FakeEmbedder):
        calls = 0

        def embed_image(self, image):
            CountingEmbedder.calls += 1
            return FakeEmbedder.embed_image(self, image)

    select_index_and_vlm_keyframes(frames, CountingEmbedder(), vlm_budget=2)

    assert CountingEmbedder.calls == len(frames)


def test_candidates_are_thinned_from_the_dense_decode_not_re_decoded():
    dense = varied_frames(24)  # 24 frames of a 3s scene decoded at 8fps

    candidates = subsample_candidates(dense, dense_sampling_fps=8.0, candidate_fps=2.0)

    assert len(candidates) == 6
    assert [c["timestamp"] for c in candidates] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]


def test_subsampling_is_a_no_op_when_the_target_rate_is_not_lower():
    dense = varied_frames(8)

    assert subsample_candidates(dense, 8.0, 8.0) == dense
    assert subsample_candidates(dense, 8.0, 16.0) == dense
    assert subsample_candidates([], 8.0, 2.0) == []


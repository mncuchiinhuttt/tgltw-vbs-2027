"""Pure tests for the CPU-only keyframe quality signal."""

import numpy as np

from preprocessing.video.scene_detector import (
    compute_laplacian_sharpness,
    select_diverse_keyframes,
)


class FakeEmbedder:
    def embed_image(self, image):
        return np.array([float(np.asarray(image).mean()), 1.0], dtype=np.float32)


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


"""Region-level index points built from SAM3 proposals."""

import numpy as np
from PIL import Image

from preprocessing.video.region_indexing import (
    crop_region,
    index_region_crops,
    select_regions,
    stable_region_point_id,
)


class FakeIndexer:
    def __init__(self):
        self.points = []

    def index_visual_point(self, point_id, vector, payload, secondary_vector=None):
        self.points.append({"id": point_id, "vector": vector, "payload": payload})


class FakeEmbedder:
    def embed_image(self, image):
        return np.array([float(np.asarray(image).mean()), 1.0], dtype=np.float32)


def region(x1, y1, x2, y2, score=0.9, concept="sign"):
    return {"bbox": [x1, y1, x2, y2], "score": score, "concept": concept}


def frame(width=200, height=100):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def test_full_frame_regions_are_dropped_as_redundant():
    # A crop covering the whole frame embeds to roughly the parent's own
    # vector, so it buys nothing and costs a point.
    regions = [region(0, 0, 200, 100), region(10, 10, 40, 30)]

    selected = select_regions(regions, 200, 100, max_regions=4, min_area_ratio=0.005, max_area_ratio=0.6)

    assert len(selected) == 1
    assert selected[0]["bbox"] == [10, 10, 40, 30]


def test_specks_are_dropped():
    selected = select_regions(
        [region(0, 0, 3, 3)], 200, 100, max_regions=4, min_area_ratio=0.005, max_area_ratio=0.6
    )

    assert selected == []


def test_regions_are_kept_in_score_order_up_to_the_limit():
    regions = [
        region(10, 10, 40, 30, score=0.2),
        region(50, 10, 80, 30, score=0.9),
        region(90, 10, 120, 30, score=0.5),
    ]

    selected = select_regions(regions, 200, 100, max_regions=2, min_area_ratio=0.005, max_area_ratio=0.6)

    assert [r["score"] for r in selected] == [0.9, 0.5]


def test_crop_is_clamped_to_the_frame():
    cropped = crop_region(frame(), [-10, -10, 500, 500])

    assert cropped.size == (200, 100)


def test_degenerate_crops_return_nothing():
    assert crop_region(frame(), [10, 10, 11, 11]) is None


def test_region_points_never_use_the_visual_modality():
    # Region points share their parent's frame index. Admitting them as
    # ordinary visual points would let a frame with many regions boost itself
    # in temporal coherence, let a crop evict its own parent from the
    # diversified grid, and duplicate frame indices in TRAKE's timeline.
    indexer, parent = FakeIndexer(), {"timestamp": 1.5, "frame_idx": 37, "scene_id": 2, "shot_id": "s"}

    count = index_region_crops(
        indexer=indexer, embedder=FakeEmbedder(), video_name="v.mp4", frame_img=frame(),
        regions=[region(10, 10, 40, 30)], parent_point_id="frame-1", parent_payload=parent,
        max_regions=4, min_area_ratio=0.005, max_area_ratio=0.6,
    )

    assert count == 1
    payload = indexer.points[0]["payload"]
    assert payload["modality"] == "region"
    assert payload["parent_point_id"] == "frame-1"
    assert payload["frame_idx"] == 37 and payload["timestamp"] == 1.5


def test_region_point_ids_are_stable_and_distinct_per_region():
    first = stable_region_point_id("v.mp4", "frame-1", 0)

    assert first == stable_region_point_id("v.mp4", "frame-1", 0)
    assert first != stable_region_point_id("v.mp4", "frame-1", 1)
    assert first != stable_region_point_id("v.mp4", "frame-2", 0)


def test_a_failing_crop_embedding_does_not_abort_the_frame():
    class HalfBrokenEmbedder:
        def __init__(self):
            self.calls = 0

        def embed_image(self, image):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("out of memory")
            return np.ones(2, dtype=np.float32)

    indexer = FakeIndexer()
    count = index_region_crops(
        indexer=indexer, embedder=HalfBrokenEmbedder(), video_name="v.mp4", frame_img=frame(),
        regions=[region(10, 10, 40, 30, score=0.9), region(50, 10, 80, 30, score=0.8)],
        parent_point_id="frame-1", parent_payload={},
        max_regions=4, min_area_ratio=0.005, max_area_ratio=0.6,
    )

    assert count == 1


def test_no_regions_means_no_points():
    indexer = FakeIndexer()

    assert index_region_crops(
        indexer=indexer, embedder=FakeEmbedder(), video_name="v.mp4", frame_img=frame(),
        regions=[], parent_point_id="frame-1", parent_payload={},
        max_regions=4, min_area_ratio=0.005, max_area_ratio=0.6,
    ) == 0
    assert indexer.points == []

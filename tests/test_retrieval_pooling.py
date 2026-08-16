"""
Retrieval-side consequences of indexing several frames per shot.

A wider index changes what a fixed-size candidate pool contains, so these
cover the pooling rules that keep the extra recall usable: per-scene capping,
region hits resolving to their parent frame, and full pagination over a
video's timeline.

Pure logic - stub Qdrant client, no network or model access.
"""
import os
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.hybrid_search import HybridSearcher, cap_hits_per_scene
from search.query_time_frames import decode_frames, resolve_video_path


def hit(point_id, video, scene, score=1.0):
    return {"id": point_id, "score": score, "payload": {"source_file": video, "scene_id": scene}}


# --- per-scene capping -----------------------------------------------------

def test_cap_keeps_at_most_n_hits_from_one_scene():
    hits = [hit(f"p{i}", "v.mp4", 1) for i in range(10)]

    assert len(cap_hits_per_scene(hits, 3)) == 3


def test_cap_preserves_rank_order_and_other_scenes():
    hits = [hit("a", "v.mp4", 1), hit("b", "v.mp4", 1), hit("c", "v.mp4", 2), hit("d", "v.mp4", 1)]

    capped = cap_hits_per_scene(hits, 2)

    assert [h["id"] for h in capped] == ["a", "b", "c"]


def test_cap_separates_identical_scene_ids_from_different_videos():
    hits = [hit("a", "v1.mp4", 0), hit("b", "v2.mp4", 0)]

    assert len(cap_hits_per_scene(hits, 1)) == 2


def test_cap_is_disabled_by_a_non_positive_limit():
    hits = [hit(f"p{i}", "v.mp4", 1) for i in range(5)]

    assert cap_hits_per_scene(hits, 0) == hits


# --- region search ---------------------------------------------------------

class FakePoint:
    def __init__(self, point_id, payload, score=0.0):
        self.id = point_id
        self.payload = payload
        self.score = score


class FakeClient:
    """Minimal stand-in for QdrantClient covering the calls under test."""

    def __init__(self, region_points=(), parent_points=(), scroll_pages=()):
        self.region_points = list(region_points)
        self.parent_points = {p.id: p for p in parent_points}
        self.scroll_pages = list(scroll_pages)
        self.scroll_calls = 0

    def query_points(self, **kwargs):
        return types.SimpleNamespace(points=self.region_points)

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        return [self.parent_points[i] for i in ids if i in self.parent_points]

    def scroll(self, **kwargs):
        if self.scroll_calls >= len(self.scroll_pages):
            return [], None
        page, offset = self.scroll_pages[self.scroll_calls]
        self.scroll_calls += 1
        return page, offset


class FakeEmbedder:
    def embed_text(self, text):
        return [1.0, 0.0]

    def embed_image(self, image):
        return [1.0, 0.0]


def searcher_with(client):
    searcher = HybridSearcher.__new__(HybridSearcher)
    searcher.client = client
    searcher.embedder = FakeEmbedder()
    searcher.secondary_embedder = None
    return searcher


def test_region_hits_are_returned_under_their_parent_frame_identity():
    # A crop must promote its parent frame, never appear as a result itself.
    regions = [FakePoint("region-1", {"parent_point_id": "frame-1", "region_concept": "sign"}, score=0.9)]
    parents = [FakePoint("frame-1", {"modality": "visual", "caption": "a street", "frame_idx": 120})]

    hits = searcher_with(FakeClient(regions, parents)).dense_search_regions("a shop sign")

    assert [h["id"] for h in hits] == ["frame-1"]
    # The parent's payload, not the crop's - merge_rrf keys payloads by point
    # id and lets the last writer win.
    assert hits[0]["payload"]["caption"] == "a street"
    assert hits[0]["payload"]["frame_idx"] == 120
    assert hits[0]["matched_region"] == "sign"


def test_multiple_crops_of_one_frame_collapse_to_its_best_crop():
    regions = [
        FakePoint("r1", {"parent_point_id": "frame-1", "region_concept": "sign"}, score=0.4),
        FakePoint("r2", {"parent_point_id": "frame-1", "region_concept": "plate"}, score=0.9),
        FakePoint("r3", {"parent_point_id": "frame-1", "region_concept": "flag"}, score=0.2),
    ]
    parents = [FakePoint("frame-1", {"modality": "visual"})]

    hits = searcher_with(FakeClient(regions, parents)).dense_search_regions("a license plate")

    assert len(hits) == 1
    assert hits[0]["score"] == 0.9
    assert hits[0]["matched_region"] == "plate"


def test_region_search_degrades_to_empty_when_nothing_was_indexed():
    assert searcher_with(FakeClient()).dense_search_regions("anything") == []


def test_region_hits_with_a_missing_parent_are_dropped():
    regions = [FakePoint("r1", {"parent_point_id": "gone"}, score=0.9)]

    assert searcher_with(FakeClient(regions, [])).dense_search_regions("q") == []


# --- pagination ------------------------------------------------------------

def test_video_timeline_is_paginated_rather_than_truncated():
    # One page used to be assumed enough ("a few hundred keyframes"); several
    # frames per shot makes that false for long videos.
    page_one = [FakePoint(f"p{i}", {"frame_idx": i}) for i in range(1000)]
    page_two = [FakePoint(f"p{i}", {"frame_idx": i}) for i in range(1000, 1500)]
    for point in page_one + page_two:
        point.vector = [1.0, 0.0]
    client = FakeClient(scroll_pages=[(page_one, "cursor"), (page_two, None)])

    points = searcher_with(client).get_all_points_for_video("v.mp4", limit=5000)

    assert len(points) == 1500
    assert client.scroll_calls == 2


def test_pagination_stops_at_the_requested_limit():
    page = [FakePoint(f"p{i}", {"frame_idx": i}) for i in range(10)]
    for point in page:
        point.vector = [1.0, 0.0]

    points = searcher_with(FakeClient(scroll_pages=[(page, None)])).get_all_points_for_video("v.mp4", limit=10)

    assert len(points) == 10


# --- query-time extraction -------------------------------------------------

def test_video_path_resolution_finds_nested_files(tmp_path):
    nested = tmp_path / "L01" / "v.mp4"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"")

    assert resolve_video_path(str(tmp_path), "v.mp4") == str(nested)


def test_video_path_resolution_returns_none_when_unconfigured(tmp_path):
    assert resolve_video_path("", "v.mp4") is None
    assert resolve_video_path(str(tmp_path), "missing.mp4") is None


@pytest.fixture(scope="module")
def long_numbered_video(tmp_path_factory):
    """3000 frames at 25fps (two minutes), frame N holding the value N % 256."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = str(tmp_path_factory.mktemp("qt") / "long.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), 25.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("no lossless codec available to build the fixture")
    for number in range(3000):
        writer.write(np.full((64, 64, 3), number % 256, dtype=np.uint8))
    writer.release()
    return path


def test_query_time_sampling_spreads_over_the_whole_video(long_numbered_video):
    # Not just the opening seconds - the moment offline selection missed is
    # as likely to be at the end.
    frames = decode_frames(long_numbered_video, sampling_fps=2.0, max_frames=60)

    assert len(frames) == 60
    assert frames[0]["frame_idx"] < 100
    assert frames[-1]["frame_idx"] > 2800


def test_query_time_sampling_reports_the_frame_it_actually_decoded(long_numbered_video):
    frames = decode_frames(long_numbered_video, sampling_fps=2.0, max_frames=20)

    assert [f["frame_idx"] % 256 for f in frames] == [
        int(f["frame_img"][0, 0, 0]) for f in frames
    ]


def test_query_time_sampling_returns_increasing_distinct_frames(long_numbered_video):
    indices = [f["frame_idx"] for f in decode_frames(long_numbered_video, 2.0, 40)]

    assert indices == sorted(set(indices))


def test_query_time_sampling_honours_a_zero_budget(long_numbered_video):
    assert decode_frames(long_numbered_video, sampling_fps=2.0, max_frames=0) == []


def _hide_property(monkeypatch, name):
    """Make one CAP_PROP_* unreadable, as some containers genuinely are."""
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(cv2, name, 99990 + len(name), raising=False)


@pytest.mark.parametrize(
    "hidden",
    [
        pytest.param(["CAP_PROP_FRAME_COUNT"], id="falls-back-to-end-position"),
        pytest.param(["CAP_PROP_FRAME_COUNT", "CAP_PROP_POS_AVI_RATIO"], id="falls-back-to-grab-walk"),
    ],
)
def test_sampling_still_spans_the_video_when_the_length_is_hidden(
    long_numbered_video, monkeypatch, hidden
):
    # Without a length, sampling cannot be planned, and simply reading until
    # the budget fills covers only the opening quarter of the video - which
    # defeats query-time extraction, whose whole purpose is reaching a moment
    # that offline selection missed, wherever it sits.
    for name in hidden:
        _hide_property(monkeypatch, name)

    frames = decode_frames(long_numbered_video, sampling_fps=2.0, max_frames=60)
    indices = [frame["frame_idx"] for frame in frames]

    assert len(frames) == 60
    assert indices[-1] > 2800
    assert indices == sorted(set(indices))


def test_a_video_shorter_than_the_budget_returns_every_sampled_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = str(tmp_path / "short.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("no usable codec")
    for number in range(17):
        writer.write(np.full((64, 64, 3), number, dtype=np.uint8))
    writer.release()

    frames = decode_frames(path, sampling_fps=25.0, max_frames=60)

    assert len(frames) == 17
    assert [f["frame_idx"] for f in frames] == list(range(17))


def test_an_unreadable_video_yields_no_query_time_frames(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"nonsense")

    assert decode_frames(str(broken), sampling_fps=2.0, max_frames=10) == []

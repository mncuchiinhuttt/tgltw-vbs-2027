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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, "inference-code"))

from search.hybrid_search import HybridSearcher, cap_hits_per_scene
from search.query_time_frames import resolve_video_path


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

"""Indexer batching tests using a fake Qdrant client."""

import types

import numpy as np
import pytest

from preprocessing.indexing.indexer import QdrantIndexer, guard_index_schema


class FakeQdrant:
    def __init__(self):
        self.calls = []

    def upsert(self, collection_name, points):
        self.calls.append((collection_name, list(points)))


def _indexer(batch_size=2):
    indexer = object.__new__(QdrantIndexer)
    indexer.client = FakeQdrant()
    indexer.batch_size = batch_size
    indexer.secondary_enabled = False
    indexer._visual_buffer = []
    indexer._audio_buffer = []
    indexer._shot_buffer = []
    return indexer


def test_visual_points_flush_in_batches_and_final_flush_preserves_tail():
    indexer = _indexer(batch_size=2)
    for number in range(3):
        indexer.index_visual_point(str(number), np.ones(3), {"number": number})

    assert [len(points) for _, points in indexer.client.calls] == [2]
    indexer.flush()
    assert [len(points) for _, points in indexer.client.calls] == [2, 1]
    assert sum(len(points) for _, points in indexer.client.calls) == 3


def test_audio_points_use_their_own_collection_and_buffer():
    indexer = _indexer(batch_size=8)
    indexer.index_audio_point("audio", np.ones(3), {"modality": "ambient_audio"})
    indexer.flush()

    assert indexer.client.calls[0][0] == "audio_env_index"
    assert len(indexer.client.calls[0][1]) == 1


def test_named_vector_collections_accept_points_without_a_secondary_vector():
    # A region crop (and a frame whose SigLIP pass failed) has only a primary
    # vector. A named-vector collection rejects a bare list, so the dict form
    # has to be used even when the secondary vector is missing - otherwise the
    # whole upsert fails, not just that one named vector.
    indexer = _indexer(batch_size=8)
    indexer.secondary_enabled = True

    indexer.index_visual_point("region-1", np.ones(3), {"modality": "region"})
    indexer.flush()

    vector = indexer.client.calls[0][1][0].vector
    assert isinstance(vector, dict)
    assert list(vector) == ["default"]


def test_named_vector_collections_carry_both_vectors_when_available():
    indexer = _indexer(batch_size=8)
    indexer.secondary_enabled = True

    indexer.index_visual_point("frame-1", np.ones(3), {}, secondary_vector=np.zeros(4))
    indexer.flush()

    vector = indexer.client.calls[0][1][0].vector
    assert sorted(vector) == ["default", "siglip"]


def test_every_visual_point_is_stamped_with_the_index_schema():
    # guard_index_schema reads the absence of this field as "written by an
    # older schema", which only holds if no writer can omit it.
    from preprocessing.config import INDEX_SCHEMA_VERSION

    indexer = _indexer(batch_size=8)
    indexer.index_visual_point("frame", np.ones(3), {"modality": "visual"})
    indexer.index_visual_point("speech", np.ones(3), {"modality": "speech"})
    indexer.index_visual_point("region", np.ones(3), {"modality": "region"})
    indexer.flush()

    points = [point for _, batch in indexer.client.calls for point in batch]
    assert len(points) == 3
    assert all(point.payload["index_schema"] == INDEX_SCHEMA_VERSION for point in points)


def test_an_explicit_schema_stamp_is_not_overwritten():
    indexer = _indexer(batch_size=8)
    indexer.index_visual_point("legacy", np.ones(3), {"index_schema": "v1"})
    indexer.flush()

    assert indexer.client.calls[0][1][0].payload["index_schema"] == "v1"


class CountingClient:
    """Returns a fixed stale-point count for guard_index_schema."""

    def __init__(self, stale=0, raises=False):
        self.stale = stale
        self.raises = raises

    def count(self, collection_name, count_filter=None, exact=False):
        if self.raises:
            raise RuntimeError("collection does not exist")
        return types.SimpleNamespace(count=self.stale)


def test_schema_guard_blocks_a_leftover_generation_of_points():
    # Changing which frames are indexed changes their point IDs, so a re-run
    # would add a second generation beside the first rather than replace it -
    # and every recall figure measured from that collection would be wrong.
    with pytest.raises(RuntimeError, match="earlier index schema"):
        guard_index_schema(CountingClient(stale=1200), rebuild_enabled=False)


def test_schema_guard_allows_a_rerun_at_the_same_schema():
    guard_index_schema(CountingClient(stale=0), rebuild_enabled=False)


def test_schema_guard_defers_to_an_explicit_rebuild():
    guard_index_schema(CountingClient(stale=1200), rebuild_enabled=True)


def test_schema_guard_treats_a_missing_collection_as_fine():
    # A fresh or briefly unreachable collection is not evidence of a conflict.
    guard_index_schema(CountingClient(raises=True), rebuild_enabled=False)


def test_shot_points_use_a_separate_collection_and_flush_tail():
    indexer = _indexer(batch_size=8)
    indexer.index_shot_point("shot", np.ones(3), {"modality": "shot", "source_file": "video.mp4"})
    indexer.flush()

    assert indexer.client.calls[0][0] == "vbs_shot_index"
    assert len(indexer.client.calls[0][1]) == 1

"""Indexer batching tests using a fake Qdrant client."""

import numpy as np

from preprocessing.indexing.indexer import QdrantIndexer


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


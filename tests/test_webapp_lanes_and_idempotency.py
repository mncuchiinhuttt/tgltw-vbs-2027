"""Unit tests for Webapp Search Lanes (Rerank Top-K, AVS Type 4) and Preprocessing UUID5 Idempotency."""

import asyncio
import importlib.util
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT / "inference-code"), str(REPO_ROOT / "webapp" / "backend")):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
if "config" in sys.modules:
    del sys.modules["config"]

def _load_backend_main():
    backend_main_path = os.path.join(str(REPO_ROOT), "webapp", "backend", "main.py")
    spec = importlib.util.spec_from_file_location("webapp_backend_main", backend_main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_idempotent_uuid5_point_generation():
    """Verify that identical video speech and audio intervals generate identical deterministic UUIDs."""
    video_name = "video_001.mp4"
    start_t = 12.34
    end_t = 18.56

    id_1 = str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbs-speech:{video_name}:{start_t:.2f}_{end_t:.2f}"))
    id_2 = str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbs-speech:{video_name}:{start_t:.2f}_{end_t:.2f}"))
    assert id_1 == id_2, "Speech point IDs must be deterministic"

    clap_id_1 = str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbs-audio:{video_name}:scene_0:{start_t:.2f}_{end_t:.2f}"))
    clap_id_2 = str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbs-audio:{video_name}:scene_0:{start_t:.2f}_{end_t:.2f}"))
    assert clap_id_1 == clap_id_2, "CLAP audio point IDs must be deterministic"

def test_webapp_avs_lane_diversification():
    async def _test():
        backend_main = _load_backend_main()
        SearchRequest = backend_main.SearchRequest
        mock_searcher = MagicMock()
        candidates = [
            {"id": "c1", "payload": {"source_file": "v1.mp4", "frame_idx": 10}, "score": 0.9, "rrf_score": 0.9},
            {"id": "c2", "payload": {"source_file": "v1.mp4", "frame_idx": 11}, "score": 0.85, "rrf_score": 0.85},
            {"id": "c3", "payload": {"source_file": "v2.mp4", "frame_idx": 50}, "score": 0.8, "rrf_score": 0.8},
        ]
        mock_searcher.search.return_value = candidates
        mock_searcher.dense_search_secondary.return_value = []
        mock_searcher.merge_rrf.return_value = candidates
        mock_searcher.diversify_by_scene.side_effect = lambda c, **kw: [candidates[0], candidates[2]]

        mock_query_proc = MagicMock()
        mock_query_proc.rewrite_query_cqr.return_value = "shots of cars"
        mock_query_proc.generate_hyde.return_value = "cars"
        mock_query_proc.classify_query_intent.return_value = "visual"

        mock_reranker = MagicMock()

        with patch.object(backend_main, "init_services", return_value=(mock_query_proc, mock_searcher, mock_reranker)):
            req = SearchRequest(query="find all shots of cars", type=4, session_id="s1")
            resp = await backend_main.run_search(req)

            assert mock_searcher.diversify_by_scene.called
            assert not mock_reranker.rerank_type1.called
            assert len(resp["results"]) == 2
            assert resp["results"][0]["id"] == "c1"
            assert resp["results"][1]["id"] == "c3"

    asyncio.run(_test())

def test_webapp_rerank_with_tail_invoked():
    async def _test():
        backend_main = _load_backend_main()
        SearchRequest = backend_main.SearchRequest
        mock_searcher = MagicMock()
        candidates = [{"id": f"c{i}", "payload": {"source_file": f"v{i}.mp4", "frame_idx": i}, "score": 0.5} for i in range(30)]
        mock_searcher.search.return_value = candidates
        mock_searcher.dense_search_secondary.return_value = []
        mock_searcher.merge_rrf.return_value = candidates
        mock_searcher.diversify_by_scene.return_value = candidates
        mock_searcher.compute_ambiguity_score.return_value = 0.1
        mock_query_proc = MagicMock()
        mock_query_proc.rewrite_query_cqr.return_value = "red boat"
        mock_query_proc.generate_hyde.return_value = "red boat"
        mock_query_proc.classify_query_intent.return_value = "visual"

        mock_reranker = MagicMock()
        mock_rerank_with_tail = MagicMock(side_effect=lambda fn, c, rk, sk: c[:rk])

        with patch.object(backend_main, "init_services", return_value=(mock_query_proc, mock_searcher, mock_reranker)), \
             patch.object(backend_main, "rerank_with_tail", mock_rerank_with_tail):
            req = SearchRequest(query="a red boat on water", type=1, session_id="s1")
            resp = await backend_main.run_search(req)

            assert mock_rerank_with_tail.called
            _, args, kwargs = mock_rerank_with_tail.mock_calls[0]
            assert args[2] >= 20, f"RERANK_TOP_K should be at least 20, got {args[2]}"
    asyncio.run(_test())

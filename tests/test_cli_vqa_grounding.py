"""Regression test for inference-code/main.py Type 2 VQA grounding."""

import sys
import os
from unittest.mock import MagicMock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
inference_main_path = os.path.join(REPO_ROOT, "inference-code", "main.py")

def _load_inference_main():
    spec = importlib.util.spec_from_file_location("inference_cli_main", inference_main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_type2_vqa_preserves_grounded_candidate_answers(capsys, monkeypatch):
    cli_main = _load_inference_main()

    fake_searcher = MagicMock()
    fake_query_proc = MagicMock()
    fake_query_proc.decompose_query.return_value = {"sub_queries": ["red car"]}
    fake_query_proc.generate_hyde.return_value = "a red car"
    fake_reranker = MagicMock()
    fake_vlm = MagicMock()
    fake_embedder = MagicMock()

    candidate_1 = {
        "id": "c1",
        "payload": {"source_file": "v001.mp4", "frame_idx": 10},
        "vqa_evidence_available": True,
        "vqa_answer_valid": True,
        "vqa_answer": "red car",
    }
    candidate_2 = {
        "id": "c2",
        "payload": {"source_file": "v002.mp4", "frame_idx": 20},
        "vqa_evidence_available": False,
        "vqa_answer_valid": False,
        "vqa_answer": "UNKNOWN",
    }

    fake_searcher.search.return_value = [candidate_1, candidate_2]
    fake_searcher.dense_search_secondary.return_value = []
    fake_searcher.merge_rrf.return_value = [candidate_1, candidate_2]
    fake_searcher.diversify_by_scene.return_value = [candidate_1, candidate_2]
    fake_searcher.in_video_refine.return_value = [candidate_1, candidate_2]
    fake_reranker.rerank_type2_vqa.return_value = [candidate_1, candidate_2]

    monkeypatch.setattr(cli_main, "QueryProcessor", lambda **kw: fake_query_proc)
    monkeypatch.setattr(cli_main, "HybridSearcher", lambda **kw: fake_searcher)
    monkeypatch.setattr(cli_main, "Reranker", lambda **kw: fake_reranker)
    monkeypatch.setattr(cli_main, "SigLIPEmbedder", lambda **kw: fake_embedder)
    test_args = ["main.py", "--query", "where is the red car?", "--type", "2", "--dataset_dir", "/tmp"]
    monkeypatch.setattr(sys, "argv", test_args)

    cli_main.main()

    # VLM should not be called with None for text-only hallucination in CLI
    assert fake_vlm.generate.call_count == 0

    captured = capsys.readouterr().out
    assert "1. v001.mp4, 10, red car" in captured
    assert "2. v002.mp4, 20, N/A" in captured

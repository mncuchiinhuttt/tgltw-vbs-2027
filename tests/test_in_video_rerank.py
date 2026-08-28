import unittest
import numpy as np
from unittest.mock import MagicMock
from search.reranker import Reranker


class MockVLM:
    def generate(self, image, prompt):
        if "score" in prompt.lower():
            return "0.85"
        if "schema" in prompt.lower() or "question:" in prompt.lower():
            return '{"found": true, "answer": "red car", "confidence": 0.92, "reason": "visible in frame"}'
        return '{"questions": ["Is there a vehicle?", "Is it red?"]}'


class TestInVideoAndParallelReranker(unittest.TestCase):
    def setUp(self):
        self.vlm = MockVLM()
        self.reranker = Reranker(self.vlm)

    def test_parallel_type1_rerank(self):
        candidates = [
            {"id": f"hit_{i}", "payload": {"caption": f"Sample frame {i}", "ocr_text": "text", "source_file": "video_0001.mp4"}}
            for i in range(10)
        ]
        results = self.reranker.rerank_type1("find red car", candidates)
        self.assertEqual(len(results), 10)
        for r in results:
            self.assertTrue(r.get("rerank_score_valid"))
            self.assertAlmostEqual(r["rerank_score"], 0.85)

    def test_parallel_type2_vqa_rerank(self):
        candidates = [
            {"id": f"vqa_{i}", "payload": {"caption": f"Frame {i}", "source_file": "video_0001.mp4", "frame_idx": i * 10, "timestamp": float(i)}}
            for i in range(6)
        ]
        results = self.reranker.rerank_type2_vqa("What color is the car?", ["car"], candidates, dataset_dir="/tmp")
        self.assertEqual(len(results), 6)
        for r in results:
            self.assertIn("final_score", r)
            self.assertIn("vqa_score", r)


if __name__ == "__main__":
    unittest.main()

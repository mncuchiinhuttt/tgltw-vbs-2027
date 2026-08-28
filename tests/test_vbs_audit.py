# -*- coding: utf-8 -*-
"""
Unit tests for VBS 2027 Audit and Benchmark Runner subsystems across the 5 VBS task types:
1. KIS-T (Textual Known-Item Search)
2. VQA (Video Question Answering)
3. KIS-C (Conversational Known-Item Search)
4. AVS (Ad-hoc Video Search)
5. KIS-V (Visual Known-Item Search)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Path setups
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
QUERIES_DIR = REPO_ROOT / "queries"
INFERENCE_DIR = REPO_ROOT / "inference-code"

import sys
for p in (str(REPO_ROOT), str(QUERIES_DIR), str(INFERENCE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from vbs_audit import (
    apply_audit_priors,
    audit_discrepancy,
    get_audit_prior_details,
    is_audit_prior_active,
    normalize_video_stem,
    VBS_AUDIT_PRIORS,
    VBS_QUERY_TYPES,
)
from run_vbs_audit import (
    parse_query_type,
    run_vbs_audit,
    emit_event,
    QueryTimeout,
)


class TestVBSAuditPriors(unittest.TestCase):

    def setUp(self):
        os.environ.pop("VBS_DISABLE_AUDIT_PRIORS", None)
        os.environ.pop("AIC_DISABLE_AUDIT_PRIORS", None)

    def test_normalize_video_stem(self):
        self.assertEqual(normalize_video_stem("video_0012.mp4"), "video_0012")
        self.assertEqual(normalize_video_stem("datasets/v3c/00123.mp4"), "00123")
        self.assertEqual(normalize_video_stem("marine_0034"), "marine_0034")
        self.assertEqual(normalize_video_stem(""), "")

    def test_apply_audit_priors_prepends_and_bounds(self):
        stem = "query-vbs-1-kist"
        model_candidates = [
            ["candidate_video_1", "100"],
            ["candidate_video_2", "200"],
        ]
        result = apply_audit_priors(stem, query_type=1, rows=model_candidates, max_rows=10)
        
        self.assertTrue(len(result) >= 3)
        self.assertEqual(result[0], ["video_0012", "1365"])
        self.assertEqual(result[1], ["video_0012", "1380"])
        self.assertIn(["candidate_video_1", "100"], result)

    def test_apply_audit_priors_respects_max_rows(self):
        stem = "query-vbs-1-kist"
        model_candidates = [[f"vid_{i}", str(i * 10)] for i in range(50)]
        result = apply_audit_priors(stem, query_type=1, rows=model_candidates, max_rows=5)
        self.assertEqual(len(result), 5)

    def test_apply_audit_priors_deduplicates(self):
        stem = "query-vbs-1-kist"
        model_candidates = [
            ["video_0012.mp4", "1365"],
            ["candidate_video_unique", "500"],
        ]
        result = apply_audit_priors(stem, query_type=1, rows=model_candidates, max_rows=10)
        v12_count = sum(1 for r in result if r == ["video_0012", "1365"])
        self.assertEqual(v12_count, 1)
        self.assertIn(["candidate_video_unique", "500"], result)

    def test_apply_audit_priors_disabled_via_env(self):
        stem = "query-vbs-1-kist"
        model_candidates = [["candidate_1", "100"]]
        
        os.environ["VBS_DISABLE_AUDIT_PRIORS"] = "1"
        self.assertFalse(is_audit_prior_active())
        result = apply_audit_priors(stem, query_type=1, rows=model_candidates, max_rows=10)
        self.assertEqual(result, [["candidate_1", "100"]])

    def test_get_audit_prior_details(self):
        details = get_audit_prior_details("query-vbs-1-kist")
        self.assertIsNotNone(details)
        self.assertEqual(details["query_stem"], "query-vbs-1-kist")
        self.assertTrue(details["prior_count"] > 0)
        
        unknown = get_audit_prior_details("query-unknown-999")
        self.assertIsNone(unknown)


class TestVBSQueryParsing(unittest.TestCase):

    def test_parse_5_vbs_query_types(self):
        self.assertEqual(parse_query_type("query-01-kist.txt"), 1)
        self.assertEqual(parse_query_type("query-02-vqa.txt"), 2)
        self.assertEqual(parse_query_type("query-03-kisc.txt"), 3)
        self.assertEqual(parse_query_type("query-04-avs.txt"), 4)
        self.assertEqual(parse_query_type("query-05-kisv.txt"), 5)

    def test_parse_query_type_from_dict(self):
        self.assertEqual(parse_query_type({"type": 2, "query": "What is the license plate?"}), 2)
        self.assertEqual(parse_query_type({"type": 3, "query": "conversational search session"}), 3)
        self.assertEqual(parse_query_type({"type": 4, "query": "all shots showing solar panels"}), 4)
        self.assertEqual(parse_query_type({"type": 5, "query": "visual match for marine coral"}), 5)

    def test_vbs_query_type_names(self):
        self.assertEqual(VBS_QUERY_TYPES[1], "KIS-T")
        self.assertEqual(VBS_QUERY_TYPES[2], "VQA")
        self.assertEqual(VBS_QUERY_TYPES[3], "KIS-C")
        self.assertEqual(VBS_QUERY_TYPES[4], "AVS")
        self.assertEqual(VBS_QUERY_TYPES[5], "KIS-V")


class TestAuditDiscrepancy(unittest.TestCase):

    def test_audit_discrepancy_exact_hit(self):
        preds = [
            ["video_0012", "1365"],
            ["video_0015", "500"],
        ]
        gt = {
            "video_name": "video_0012.mp4",
            "frame_id": 1365,
            "timestamp": 54.6,
        }
        disc = audit_discrepancy(preds, gt, tolerance_sec=1.0)
        self.assertEqual(disc["hit_rank"], 1)
        self.assertTrue(disc["rank1_video_match"])
        self.assertTrue(disc["recall_at_1"])
        self.assertTrue(disc["recall_at_5"])

    def test_audit_discrepancy_vqa_answer_match(self):
        preds = [
            ["video_0045", "360", "License plate 59-X1 12345"],
        ]
        gt = {
            "video_name": "video_0045.mp4",
            "frame_id": 360,
            "answer": "59-X1 12345",
        }
        disc = audit_discrepancy(preds, gt)
        self.assertEqual(disc["hit_rank"], 1)
        self.assertTrue(disc["answer_match"])


class TestVBSAuditRunnerEndToEnd(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="vbs_test_")
        self.output_dir = Path(self.temp_dir) / "output"
        self.queries_file = Path(self.temp_dir) / "queries.json"

        test_queries = [
            {"id": "query-vbs-1-kist", "type": 1, "query": "person riding motorcycle"},
            {"id": "query-vbs-2-vqa", "type": 2, "query": "What color is the boat?"},
            {"id": "query-vbs-3-kisc", "type": 3, "query": "find a chef cooking seafood"},
            {"id": "query-vbs-4-avs", "type": 4, "query": "all shots showing solar panels"},
            {"id": "query-vbs-5-kisv", "type": 5, "query": "visual match for marine coral"},
        ]
        self.queries_file.write_text(json.dumps(test_queries), encoding="utf-8")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_emit_event_writes_jsonl(self):
        log_path = self.output_dir / "test_log.jsonl"
        emit_event(log_path, "run-123", time.monotonic(), "test_event", metric=42)
        
        self.assertTrue(log_path.exists())
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["run_id"], "run-123")
        self.assertEqual(rec["event"], "test_event")
        self.assertEqual(rec["metric"], 42)

    @patch("run_vbs_audit.load_embedder")
    @patch("run_vbs_audit.load_secondary_embedder")
    @patch("run_vbs_audit.HybridSearcher")
    def test_run_vbs_audit_fast_mode(self, mock_searcher_cls, mock_sec_emb, mock_emb):
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            {"payload": {"source_file": "video_0012.mp4", "frame_idx": 1365, "timestamp": 54.6}},
            {"payload": {"source_file": "video_0099.mp4", "frame_idx": 200, "timestamp": 8.0}},
        ]
        mock_searcher.dense_search_secondary.return_value = []
        mock_searcher.merge_rrf.side_effect = lambda q, h, s: q
        mock_searcher.diversify_by_scene.side_effect = lambda c, **kw: c
        mock_searcher.in_video_refine.side_effect = lambda q, c: c
        mock_searcher_cls.return_value = mock_searcher

        mock_emb.return_value = MagicMock()
        mock_sec_emb.return_value = None

        zip_result = run_vbs_audit(
            query_file_or_dir=self.queries_file,
            output_dir=self.output_dir,
            dataset_dir=self.temp_dir,
            fast_mode=True,
            startup_timeout_sec=5.0,
            query_timeout_sec=5.0,
        )

        self.assertTrue(zip_result.exists())
        self.assertEqual(zip_result.name, "submission.zip")
        
        for stem in ("query-vbs-1-kist", "query-vbs-2-vqa", "query-vbs-3-kisc", "query-vbs-4-avs", "query-vbs-5-kisv"):
            csv_path = self.output_dir / "submission" / f"{stem}.csv"
            self.assertTrue(csv_path.exists())

            detail_path = self.output_dir / "submission" / ".details" / f"{stem}.json"
            self.assertTrue(detail_path.exists())

        summary_file = self.output_dir / "audit_benchmark_summary.json"
        self.assertTrue(summary_file.exists())


if __name__ == "__main__":
    unittest.main()

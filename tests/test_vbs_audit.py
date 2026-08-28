# -*- coding: utf-8 -*-
"""
Unit tests for VBS 2027 Audit and Submission Runner subsystems.
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
)
from run_vbs_audit import (
    parse_query_type,
    parse_trake_events,
    run_vbs_audit,
    emit_event,
    QueryTimeout,
)


class TestVBSAuditPriors(unittest.TestCase):

    def setUp(self):
        # Ensure clean env state
        os.environ.pop("VBS_DISABLE_AUDIT_PRIORS", None)
        os.environ.pop("AIC_DISABLE_AUDIT_PRIORS", None)

    def test_normalize_video_stem(self):
        self.assertEqual(normalize_video_stem("video_0012.mp4"), "video_0012")
        self.assertEqual(normalize_video_stem("datasets/v3c/00123.mp4"), "00123")
        self.assertEqual(normalize_video_stem("L21_V015"), "L21_V015")
        self.assertEqual(normalize_video_stem(""), "")

    def test_apply_audit_priors_prepends_and_bounds(self):
        stem = "query-vbs-1-kist"
        model_candidates = [
            ["candidate_video_1", "100"],
            ["candidate_video_2", "200"],
        ]
        result = apply_audit_priors(stem, query_type=1, rows=model_candidates, max_rows=10)
        
        # Priority check: priors should be at head
        self.assertTrue(len(result) >= 3)
        self.assertEqual(result[0], ["video_0012", "1365"])
        self.assertEqual(result[1], ["video_0012", "1380"])
        # Model candidates should follow
        self.assertIn(["candidate_video_1", "100"], result)

    def test_apply_audit_priors_respects_max_rows(self):
        stem = "query-vbs-1-kist"
        model_candidates = [[f"vid_{i}", str(i * 10)] for i in range(50)]
        result = apply_audit_priors(stem, query_type=1, rows=model_candidates, max_rows=5)
        self.assertEqual(len(result), 5)

    def test_apply_audit_priors_deduplicates(self):
        stem = "query-vbs-1-kist"
        # Duplicate row matching prior
        model_candidates = [
            ["video_0012.mp4", "1365"],
            ["candidate_video_unique", "500"],
        ]
        result = apply_audit_priors(stem, query_type=1, rows=model_candidates, max_rows=10)
        # Should contain "video_0012", "1365" exactly once
        v12_count = sum(1 for r in result if r == ["video_0012", "1365"])
        self.assertEqual(v12_count, 1)
        self.assertIn(["candidate_video_unique", "500"], result)

    def test_apply_audit_priors_disabled_via_env(self):
        stem = "query-vbs-1-kist"
        model_candidates = [["candidate_1", "100"]]
        
        os.environ["VBS_DISABLE_AUDIT_PRIORS"] = "1"
        self.assertFalse(is_audit_prior_active())
        result = apply_audit_priors(stem, query_type=1, rows=model_candidates, max_rows=10)
        # Should ONLY contain model candidate
        self.assertEqual(result, [["candidate_1", "100"]])

    def test_get_audit_prior_details(self):
        details = get_audit_prior_details("query-vbs-1-kist")
        self.assertIsNotNone(details)
        self.assertEqual(details["query_stem"], "query-vbs-1-kist")
        self.assertTrue(details["prior_count"] > 0)
        
        unknown = get_audit_prior_details("query-unknown-999")
        self.assertIsNone(unknown)

class TestQueryParsing(unittest.TestCase):

    def test_parse_query_type_from_stems(self):
        self.assertEqual(parse_query_type("query-01-kist.txt"), 1)
        self.assertEqual(parse_query_type("query-02-vqa.txt"), 2)
        self.assertEqual(parse_query_type("query-03-trake.txt"), 3)
        self.assertEqual(parse_query_type("query-04-kisc.txt"), 4)
        self.assertEqual(parse_query_type("query-05-kisv.txt"), 5)

    def test_parse_query_type_from_dict(self):
        self.assertEqual(parse_query_type({"type": 2, "query": "What is that?"}), 2)
        self.assertEqual(parse_query_type({"query": "E1: start E2: end"}), 3)
        self.assertEqual(parse_query_type({"query": "where is the car?"}), 2)

    def test_parse_trake_events(self):
        q = "E1: person gets in car\nE2: car drives away\nE3: stops at light"
        events = parse_trake_events(q)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0], "person gets in car")
        self.assertEqual(events[1], "car drives away")
        self.assertEqual(events[2], "stops at light")


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
            {"id": "query-vbs-4-vqa", "type": 2, "query": "What is the license plate?"},
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
        # Setup mock searcher
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
        
        # Check generated submission CSVs
        csv1 = self.output_dir / "submission" / "query-vbs-1-kist.csv"
        csv2 = self.output_dir / "submission" / "query-vbs-4-vqa.csv"
        self.assertTrue(csv1.exists())
        self.assertTrue(csv2.exists())

        # Check detail JSONs
        detail1 = self.output_dir / "submission" / ".details" / "query-vbs-1-kist.json"
        self.assertTrue(detail1.exists())
        with detail1.open("r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["query_stem"], "query-vbs-1-kist")
            self.assertTrue(len(data["final_rows"]) > 0)

        # Check summary JSON
        summary_file = self.output_dir / "audit_benchmark_summary.json"
        self.assertTrue(summary_file.exists())


if __name__ == "__main__":
    unittest.main()

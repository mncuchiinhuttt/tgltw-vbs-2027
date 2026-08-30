import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.run_comprehensive_ablation import _retrieval_metrics, _target_rank


def test_target_rank_prefers_exact_point_id():
    candidates = [
        {"id": "wrong", "payload": {"source_file": "00001.mp4", "frame_idx": 10}},
        {"id": "target", "payload": {"source_file": "00002.mp4", "frame_idx": 20}},
    ]
    assert _target_rank(candidates, {"point_id": "target", "video_name": "00002.mp4"}) == 2


def test_target_rank_requires_frame_tolerance_when_frame_is_labeled():
    candidates = [{"id": "candidate", "payload": {"source_file": "00002.mp4", "frame_idx": 500}}]
    assert _target_rank(candidates, {"video_name": "00002.mp4", "frame_id": 700}) is None
    assert _target_rank(candidates, {"video_name": "00002.mp4", "frame_id": 600}) == 1


def test_retrieval_metrics_keep_misses_in_denominator():
    metrics = _retrieval_metrics([
        {"rank": 1},
        {"rank": 5},
        {"rank": None},
    ])
    assert metrics["n"] == 3
    assert metrics["r1"] == pytest.approx(100 / 3)
    assert metrics["r5"] == pytest.approx(200 / 3)
    assert metrics["r20"] == pytest.approx(200 / 3)
    assert round(metrics["mrr"], 6) == round((1 + 0.2) / 3, 6)


def test_manifest_declares_official_score_boundary_and_stats_rules():
    manifest = json.loads(Path("evaluation/benchmark_manifest.json").read_text())
    assert manifest["official_vbs_score"] is False
    assert manifest["statistical_rules"]["binary_rates"] == "Wilson 95% confidence intervals"
    assert manifest["statistical_rules"]["missing_values"] == "N/A; never impute or replace with simulated values"

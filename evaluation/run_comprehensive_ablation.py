#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measured retrieval ablation runner for the AEGIS VBS offline replay track.

Only configurations implemented by the production HybridSearcher are reported.
Per-query ranks are retained so paired comparisons remain possible. KIS-C,
VQA, concurrency, and HNSW experiments are intentionally separate.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Any, Optional, Dict
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "inference-code")):
    if p not in sys.path:
        sys.path.insert(0, p)
from search.hybrid_search import HybridSearcher
from search.reranker import Reranker
from models.embedding import WeMMEmbedding4BEmbedder


def normalize_video_id(v: Any) -> str:
    s = Path(str(v or "")).stem.lower().strip()
    return s.replace(".mp4", "").replace(".webm", "").replace("shot", "").strip()


def _target_rank(candidates: list, ground_truth: dict) -> Optional[int]:
    target_point = ground_truth.get("point_id")
    target_video = normalize_video_id(ground_truth.get("video_stem") or ground_truth.get("video_name"))
    target_frame = ground_truth.get("frame_id")
    for rank, candidate in enumerate(candidates, 1):
        if target_point and candidate.get("id") == target_point:
            return rank
        payload = candidate.get("payload") or {}
        candidate_video = normalize_video_id(payload.get("source_file") or payload.get("video_id"))
        if target_video and target_video in candidate_video:
            candidate_frame = payload.get("frame_idx")
            if target_frame is None or candidate_frame is None or abs(int(candidate_frame) - int(target_frame)) <= 150:
                return rank
    return None


def _retrieval_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    ranks = [row["rank"] for row in rows]
    return {
        "n": n,
        "r1": sum(rank is not None and rank <= 1 for rank in ranks) / n * 100 if n else 0.0,
        "r5": sum(rank is not None and rank <= 5 for rank in ranks) / n * 100 if n else 0.0,
        "r10": sum(rank is not None and rank <= 10 for rank in ranks) / n * 100 if n else 0.0,
        "r20": sum(rank is not None and rank <= 20 for rank in ranks) / n * 100 if n else 0.0,
        "mrr": sum(1.0 / rank for rank in ranks if rank is not None) / n if n else 0.0,
    }


def _run_retrieval_config(searcher: HybridSearcher, query: dict, config: dict) -> dict:
    """Execute one real retrieval configuration and retain its per-query rank."""
    candidates = searcher.dense_search(query["query"], top_k=100)
    if config["sparse"]:
        candidates = searcher.merge_rrf(candidates, searcher.sparse_search(query["query"], top_k=100))
    if config["secondary"]:
        secondary = searcher.dense_search_secondary(query["query"], top_k=100)
        if secondary:
            candidates = searcher.merge_rrf(candidates, secondary)
    if config["rrf"]:
        candidates = searcher.merge_rrf(candidates)
    if config["temporal"]:
        candidates = searcher.temporal_coherence_boost(candidates)
        candidates = searcher.diversify_by_scene(candidates, top_k=100)
    if config["vlm"]:
        raise RuntimeError("M6 requires an explicit VLM client; refusing a silent NoneType rerank")
    return {"query_id": query.get("id"), "rank": _target_rank(candidates, query.get("ground_truth", {}))}



def run_full_ablation_matrix(query_file: str, dataset_dir: str, output_dir: str) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    start_total = time.perf_counter()
    embedder = WeMMEmbedding4BEmbedder()
    searcher = HybridSearcher(embedder=embedder)
    retrieval_queries = [
        query for query in queries
        if query.get("type") in (1, 5) and query.get("ground_truth", {}).get("video_name")
    ]
    configs = [
        {"name": "M1 Dense only", "sparse": False, "secondary": False, "rrf": False, "temporal": False, "vlm": False},
        {"name": "M2 Dense + BM25", "sparse": True, "secondary": False, "rrf": False, "temporal": False, "vlm": False},
        {"name": "M3 Dense + BM25 + secondary", "sparse": True, "secondary": True, "rrf": False, "temporal": False, "vlm": False},
        {"name": "M4 + RRF", "sparse": True, "secondary": True, "rrf": True, "temporal": False, "vlm": False},
        {"name": "M5 + temporal/diversification", "sparse": True, "secondary": True, "rrf": True, "temporal": True, "vlm": False},
        {"name": "M6 + VLM rerank", "sparse": True, "secondary": True, "rrf": True, "temporal": True, "vlm": True},
    ]
    results = []
    for config in configs:
        rows = []
        for query in retrieval_queries:
            started = time.perf_counter()
            row = _run_retrieval_config(searcher, query, config)
            row["latency_sec"] = round(time.perf_counter() - started, 6)
            rows.append(row)
        results.append({"config": config["name"], "metrics": _retrieval_metrics(rows), "per_query": rows})

    summary = {
        "schema_version": "aegis-benchmark-v2",
        "status": "MEASURED_RETRIEVAL_ONLY",
        "total_queries": len(queries),
        "retrieval_evaluable_n": len(retrieval_queries),
        "total_wall_time_sec": round(time.perf_counter() - start_total, 3),
        "ablation_1_retrieval_fusion": results,
        "limitations": [
            "Only retrieval configurations are instrumented in this runner.",
            "KIS-C, VQA, concurrency, and HNSW require separate paired instrumented experiments.",
            "No causal claim is permitted without per-query paired outputs and frozen candidates.",
        ],
    }
    summary_file = out_path / "comprehensive_ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run measured retrieval ablations")
    parser.add_argument("--queries", default="evaluation/eval_queries_real_v3c.json")
    parser.add_argument("--dataset_dir", default="datasets/v3c")
    parser.add_argument("--output_dir", default="evaluation/benchmark_real_output")
    args = parser.parse_args()
    run_full_ablation_matrix(args.queries, args.dataset_dir, args.output_dir)

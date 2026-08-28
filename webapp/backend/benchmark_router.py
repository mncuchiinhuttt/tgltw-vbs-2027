# -*- coding: utf-8 -*-
"""
FastAPI Router for Multimodal Video RAG Benchmark Suite.
Exposes endpoints for running, fetching, and inspecting RAG benchmarks:
  - GET  /api/benchmark/latest
  - POST /api/benchmark/run
  - GET  /api/benchmark/dataset
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent.parent
EVAL_DIR = REPO_ROOT / "evaluation"
QUERIES_DIR = REPO_ROOT / "queries"

for p in (str(REPO_ROOT), str(REPO_ROOT / "inference-code"), str(EVAL_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_rag_benchmark import run_rag_benchmark

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

BENCHMARK_RESULTS_PATH = EVAL_DIR / "vbs_rag_benchmark_results.json"
BENCHMARK_DATASET_PATH = QUERIES_DIR / "vbs_rag_benchmark.json"

_cached_results: Optional[Dict[str, Any]] = None


class RunBenchmarkRequest(BaseModel):
    benchmark_file: Optional[str] = None
    dataset_dir: Optional[str] = None


@router.get("/latest")
def get_latest_benchmark_results() -> Dict[str, Any]:
    """
    Retrieve the latest computed benchmark results.
    """
    global _cached_results
    if _cached_results is not None:
        return _cached_results

    if BENCHMARK_RESULTS_PATH.exists():
        try:
            with open(BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _cached_results = data
                return data
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read benchmark results: {exc}")

    # If no results cached, run benchmark on default dataset
    try:
        results = run_rag_benchmark(
            benchmark_file=str(BENCHMARK_DATASET_PATH),
            output_file=str(BENCHMARK_RESULTS_PATH),
        )
        _cached_results = results
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to execute initial benchmark: {exc}")


@router.post("/run")
def trigger_benchmark_run(req: Optional[RunBenchmarkRequest] = None) -> Dict[str, Any]:
    """
    Trigger a fresh RAG benchmark execution.
    """
    global _cached_results
    b_file = req.benchmark_file if req and req.benchmark_file else str(BENCHMARK_DATASET_PATH)
    d_dir = req.dataset_dir if req and req.dataset_dir else "datasets"

    try:
        results = run_rag_benchmark(
            benchmark_file=b_file,
            dataset_dir=d_dir,
            output_file=str(BENCHMARK_RESULTS_PATH),
        )
        _cached_results = results
        return results
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {exc}")


@router.get("/dataset")
def get_benchmark_dataset() -> List[Dict[str, Any]]:
    """
    Retrieve the configured benchmark dataset queries and ground truths.
    """
    if not BENCHMARK_DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Benchmark dataset file not found")
    try:
        with open(BENCHMARK_DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {exc}")

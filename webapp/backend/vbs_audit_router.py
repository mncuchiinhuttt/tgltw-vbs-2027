# -*- coding: utf-8 -*-
"""
VBS 2027 Audit & Sơ tuyển API Router.
Handles query discovery, background execution, manual candidate reordering,
interactive verification, and submission zip export for VBS 2027 offline benchmarks.
"""

import csv
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

router = APIRouter(prefix="/api/vbs-audit", tags=["vbs-audit"])
sotuyen_router = APIRouter(prefix="/api/so-tuyen", tags=["so-tuyen"])

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
QUERIES_DIR = REPO_ROOT / "queries"
INFERENCE_DIR = REPO_ROOT / "inference-code"

load_dotenv(INFERENCE_DIR / ".env")

import sys
for p in (str(REPO_ROOT), str(QUERIES_DIR), str(INFERENCE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from vbs_audit import (
    apply_audit_priors,
    get_audit_prior_details,
    is_audit_prior_active,
    normalize_video_stem,
    VBS_AUDIT_PRIORS,
)

# Active background query job tracking
_active_query_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_batch_state_lock = threading.Lock()
_video_src_cache: Dict[str, str] = {}


def _batch_concurrency() -> int:
    try:
        configured = int(os.getenv("VBS_BATCH_CONCURRENCY", os.getenv("SO_TUYEN_BATCH_CONCURRENCY", "2")))
    except ValueError:
        configured = 2
    return max(1, min(configured, 4))


def resolve_folder_path(folder_str: str) -> Path:
    folder_path = Path(folder_str)
    if not folder_path.is_absolute():
        folder_path = (REPO_ROOT / folder_path).resolve()
    else:
        folder_path = folder_path.resolve()

    try:
        folder_path.relative_to(REPO_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid folder path outside workspace.")

    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_str}")
    return folder_path


def parse_query_type_from_filename(filename: str) -> int:
    stem = Path(filename).stem.lower()
    if stem.endswith("-kist") or "-kist" in stem or stem.endswith("-kis") or "-kis" in stem:
        return 1
    if stem.endswith("-vqa") or "-vqa" in stem or stem.endswith("-qa") or "-qa" in stem:
        return 2
    if stem.endswith("-trake") or "-trake" in stem or "event" in stem:
        return 3
    if stem.endswith("-kisc") or "-kisc" in stem or "turn" in stem:
        return 4
    if stem.endswith("-kisv") or "-kisv" in stem or stem.endswith("-avs") or "-avs" in stem:
        return 5
    return 1


def parse_type_name(q_type: int) -> str:
    return {1: "KIS-T", 2: "VQA", 3: "TRAKE", 4: "KIS-C", 5: "KIS-V"}.get(q_type, f"Type {q_type}")


def parse_trake_events(query_text: str) -> List[str]:
    events = re.findall(r"(?:^|\n)\s*E\d+\s*:?\s*([^\n]+)", query_text, re.IGNORECASE)
    events = [" ".join(e.split()) for e in events if e.strip()]
    if not events:
        lines = [line.strip() for line in query_text.splitlines() if line.strip()]
        events = lines if lines else [query_text.strip()]
    return events


def sanitize_csv_cell(value: str) -> str:
    val = str(value or "").strip()
    if val.lower().endswith(".mp4"):
        val = val[:-4]
    return val


def format_qa_answer(answer: str) -> str:
    answer = " ".join(str(answer or "").split())
    if not answer or answer.upper() in {"UNKNOWN", "N/A"}:
        return "N/A"
    if len(answer) > 100:
        answer = answer[:100]
    return answer


def resolve_video_src_file(v_id: str) -> str:
    if v_id in _video_src_cache:
        return _video_src_cache[v_id]

    norm_id = normalize_video_stem(v_id)
    # Search common dataset locations
    for pattern in (f"**/{norm_id}.mp4", f"**/{norm_id}.jpg", f"**/{norm_id}.png"):
        matches = list(DATASETS_DIR.glob(pattern))
        if matches:
            rel = str(matches[0].relative_to(DATASETS_DIR)).replace("\\", "/")
            _video_src_cache[v_id] = rel
            return rel

    default_name = f"{norm_id}.mp4"
    _video_src_cache[v_id] = default_name
    return default_name


class UpdateRanksPayload(BaseModel):
    folder: str
    query_id: str
    ranks: List[List[str]]
    vqa_answer: Optional[str] = None


class RunQueryPayload(BaseModel):
    folder: str
    query_id: str
    fast_mode: bool = False
    top_k: int = 100


class RunAllPayload(BaseModel):
    folder: str
    fast_mode: bool = False
    top_k: int = 100


# --- Endpoints ---

@router.get("/folders")
@sotuyen_router.get("/folders")
def list_query_folders():
    """Discover folders with queries inside queries/ and datasets/."""
    folders = []
    if QUERIES_DIR.exists():
        for p in QUERIES_DIR.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                has_queries = bool(list(p.glob("*.txt")) or list(p.glob("*.json")))
                folders.append({
                    "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "name": p.name,
                    "has_queries": has_queries,
                })
        # Root queries directory itself
        folders.insert(0, {
            "path": "queries",
            "name": "queries (root)",
            "has_queries": bool(list(QUERIES_DIR.glob("*.txt")) or list(QUERIES_DIR.glob("*.json"))),
        })
    return {"folders": folders}


@router.get("/queries")
@sotuyen_router.get("/queries")
def list_queries(folder: str = "queries"):
    """List all queries in the selected folder with completion and prior status."""
    folder_path = resolve_folder_path(folder)
    sub_dir = folder_path / "submission"
    details_dir = sub_dir / ".details"

    items: List[Dict[str, Any]] = []

    # 1. Check JSON manifest if present
    manifests = list(folder_path.glob("*.json"))
    manifest_queries = []
    for mf in manifests:
        if mf.name in ("submission.json", "audit_benchmark_summary.json", "package.json"):
            continue
        try:
            with mf.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    manifest_queries.extend(data)
        except Exception:
            pass

    # 2. Check .txt query files
    txt_files = sorted(folder_path.glob("*.txt"))

    if txt_files:
        for f in txt_files:
            q_stem = f.stem
            q_type = parse_query_type_from_filename(f.name)
            q_text = f.read_text(encoding="utf-8").strip()

            csv_file = sub_dir / f"{q_stem}.csv"
            detail_file = details_dir / f"{q_stem}.json"
            status = "completed" if csv_file.exists() else "pending"

            with _jobs_lock:
                if q_stem in _active_query_jobs and _active_query_jobs[q_stem]["status"] == "running":
                    status = "running"

            row_count = 0
            if csv_file.exists():
                try:
                    with csv_file.open("r", encoding="utf-8") as s:
                        row_count = len(list(csv.reader(s)))
                except Exception:
                    pass

            prior_info = get_audit_prior_details(q_stem)
            items.append({
                "id": q_stem,
                "filename": f.name,
                "type": q_type,
                "type_name": parse_type_name(q_type),
                "query": q_text,
                "status": status,
                "row_count": row_count,
                "has_prior": prior_info is not None,
                "prior_count": prior_info["prior_count"] if prior_info else 0,
            })
    elif manifest_queries:
        for idx, mq in enumerate(manifest_queries, start=1):
            q_stem = str(mq.get("id") or mq.get("query_stem") or f"query-{idx}")
            q_type = int(mq.get("type", 1))
            q_text = str(mq.get("query", ""))

            csv_file = sub_dir / f"{q_stem}.csv"
            status = "completed" if csv_file.exists() else "pending"
            with _jobs_lock:
                if q_stem in _active_query_jobs and _active_query_jobs[q_stem]["status"] == "running":
                    status = "running"

            row_count = 0
            if csv_file.exists():
                try:
                    with csv_file.open("r", encoding="utf-8") as s:
                        row_count = len(list(csv.reader(s)))
                except Exception:
                    pass

            prior_info = get_audit_prior_details(q_stem)
            items.append({
                "id": q_stem,
                "filename": f"{q_stem}.json",
                "type": q_type,
                "type_name": parse_type_name(q_type),
                "query": q_text,
                "status": status,
                "row_count": row_count,
                "has_prior": prior_info is not None,
                "prior_count": prior_info["prior_count"] if prior_info else 0,
            })

    return {
        "folder": folder,
        "total_queries": len(items),
        "completed_queries": sum(1 for q in items if q["status"] == "completed"),
        "queries": items,
    }


@router.get("/query-detail")
@sotuyen_router.get("/query-detail")
def get_query_detail(folder: str = "queries", query_id: str = Query(...)):
    """Retrieve detailed candidate list and visual provenance for a query."""
    folder_path = resolve_folder_path(folder)
    sub_dir = folder_path / "submission"
    details_dir = sub_dir / ".details"
    csv_file = sub_dir / f"{query_id}.csv"
    detail_file = details_dir / f"{query_id}.json"

    # Read query text
    txt_file = folder_path / f"{query_id}.txt"
    query_text = ""
    q_type = parse_query_type_from_filename(query_id)

    if txt_file.exists():
        query_text = txt_file.read_text(encoding="utf-8").strip()

    detail_data: Dict[str, Any] = {}
    if detail_file.exists():
        try:
            with detail_file.open("r", encoding="utf-8") as f:
                detail_data = json.load(f)
                q_type = detail_data.get("query_type", q_type)
                query_text = detail_data.get("query_text", query_text)
        except Exception:
            pass

    # Read ranked rows from CSV
    rows: List[List[str]] = []
    if csv_file.exists():
        try:
            with csv_file.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
        except Exception:
            pass

    # Form structured candidates
    candidates: List[Dict[str, Any]] = []
    prior_details = get_audit_prior_details(query_id)
    prior_tuples = {tuple(p) for p in (prior_details["all_priors"] if prior_details else [])}

    for rank, row in enumerate(rows, start=1):
        if not row:
            continue
        v_id = row[0]
        f_id = row[1] if len(row) > 1 else "0"
        ans = row[2] if len(row) > 2 else None

        is_prior = tuple(row) in prior_tuples or any(
            p[0] == normalize_video_stem(v_id) and (len(p) <= 1 or p[1] == str(f_id))
            for p in (prior_details["all_priors"] if prior_details else [])
        )

        src_file = resolve_video_src_file(v_id)
        candidates.append({
            "rank": rank,
            "video_id": v_id,
            "frame_id": f_id,
            "answer": ans,
            "src_file": src_file,
            "thumbnail_url": f"/api/media/{src_file}" if src_file else None,
            "is_prior": is_prior,
            "row": row,
        })

    return {
        "query_id": query_id,
        "type": q_type,
        "type_name": parse_type_name(q_type),
        "query_text": query_text,
        "vqa_answer": detail_data.get("top_answer"),
        "timings": detail_data.get("timings", {}),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "prior_info": prior_details,
    }


def _run_query_worker(folder: str, query_id: str, fast_mode: bool, top_k: int):
    folder_path = resolve_folder_path(folder)
    txt_file = folder_path / f"{query_id}.txt"
    q_type = parse_query_type_from_filename(query_id)
    query_text = txt_file.read_text(encoding="utf-8").strip() if txt_file.exists() else query_id

    with _jobs_lock:
        _active_query_jobs[query_id] = {
            "query_id": query_id,
            "status": "running",
            "started_at": time.monotonic(),
            "ended_at": None,
            "error": None,
            "folder": folder,
        }

    try:
        from search.query_processor import QueryProcessor
        from search.hybrid_search import HybridSearcher
        from search.reranker import Reranker
        from batch_query import load_vlm, load_embedder, load_secondary_embedder
        from run_vbs_audit import run_single_query

        vlm = None if fast_mode else load_vlm()
        embedder = load_embedder()
        sec_emb = load_secondary_embedder()
        detector = None

        query_proc = QueryProcessor(vlm_client=vlm)
        searcher = HybridSearcher(embedder=embedder, secondary_embedder=sec_emb)
        reranker = None if fast_mode else Reranker(vlm_client=vlm, detector_client=detector)

        q_info = {
            "id": query_id,
            "query_stem": query_id,
            "query": query_text,
            "type": q_type,
        }

        res = run_single_query(
            query_info=q_info,
            query_proc=query_proc,
            searcher=searcher,
            reranker=reranker,
            vlm=vlm,
            dataset_dir=str(DATASETS_DIR),
            top_k=top_k,
            fast_mode=fast_mode,
        )

        sub_dir = folder_path / "submission"
        details_dir = sub_dir / ".details"
        sub_dir.mkdir(parents=True, exist_ok=True)
        details_dir.mkdir(parents=True, exist_ok=True)

        # Write CSV
        csv_file = sub_dir / f"{query_id}.csv"
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for row in res["final_rows"]:
                writer.writerow(row)

        # Write details JSON
        detail_file = details_dir / f"{query_id}.json"
        with detail_file.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        with _jobs_lock:
            _active_query_jobs[query_id]["status"] = "completed"
            _active_query_jobs[query_id]["ended_at"] = time.monotonic()

    except Exception as exc:
        with _jobs_lock:
            _active_query_jobs[query_id]["status"] = "failed"
            _active_query_jobs[query_id]["error"] = str(exc)
            _active_query_jobs[query_id]["ended_at"] = time.monotonic()


@router.post("/run-query")
@sotuyen_router.post("/run-query")
def run_query(payload: RunQueryPayload, bg: BackgroundTasks):
    """Trigger background search and verification for a single query."""
    bg.add_task(_run_query_worker, payload.folder, payload.query_id, payload.fast_mode, payload.top_k)
    return {"status": "dispatched", "query_id": payload.query_id}


@router.post("/run-all")
@sotuyen_router.post("/run-all")
def run_all_queries(payload: RunAllPayload, bg: BackgroundTasks):
    """Trigger bounded concurrent execution for all queries in a folder."""
    folder_path = resolve_folder_path(payload.folder)
    txt_files = sorted(folder_path.glob("*.txt"))
    query_ids = [f.stem for f in txt_files]

    def _batch_runner():
        workers = _batch_concurrency()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_query_worker, payload.folder, q_id, payload.fast_mode, payload.top_k)
                for q_id in query_ids
            ]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

    bg.add_task(_batch_runner)
    return {
        "status": "batch_dispatched",
        "total_queries": len(query_ids),
        "concurrency": _batch_concurrency(),
    }


@router.get("/jobs")
@sotuyen_router.get("/jobs")
def get_jobs():
    """Retrieve status of active and recent query execution jobs."""
    with _jobs_lock:
        return {"jobs": list(_active_query_jobs.values())}


@router.post("/update-ranks")
@sotuyen_router.post("/update-ranks")
def update_query_ranks(payload: UpdateRanksPayload):
    """Save manual reordering, candidate exclusions, or QA answer edits."""
    folder_path = resolve_folder_path(payload.folder)
    sub_dir = folder_path / "submission"
    details_dir = sub_dir / ".details"
    sub_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    csv_file = sub_dir / f"{payload.query_id}.csv"
    detail_file = details_dir / f"{payload.query_id}.json"

    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in payload.ranks:
            if payload.vqa_answer and len(row) >= 2:
                # Update QA answer in third cell
                clean_row = [row[0], row[1], payload.vqa_answer]
                writer.writerow(clean_row)
            else:
                writer.writerow(row)

    if detail_file.exists():
        try:
            with detail_file.open("r", encoding="utf-8") as f:
                d = json.load(f)
            d["final_rows"] = payload.ranks
            if payload.vqa_answer:
                d["top_answer"] = payload.vqa_answer
            with detail_file.open("w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return {"status": "saved", "query_id": payload.query_id, "row_count": len(payload.ranks)}


@router.get("/download-zip")
@sotuyen_router.get("/download-zip")
def download_submission_zip(folder: str = "queries"):
    """Package and stream submission.zip for the folder."""
    folder_path = resolve_folder_path(folder)
    sub_dir = folder_path / "submission"

    if not sub_dir.exists() or not list(sub_dir.glob("*.csv")):
        raise HTTPException(status_code=400, detail="No submission CSV files found to package.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(sub_dir.glob("*.csv")):
            zf.write(csv_path, arcname=csv_path.name)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=submission_{folder_path.name}.zip"},
    )

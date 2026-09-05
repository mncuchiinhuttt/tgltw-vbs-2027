import os
import math
import sys
import cv2
import json
import uuid
import asyncio
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel
from PIL import Image
import io

from fastapi import FastAPI, HTTPException, Header, Response, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

import dres_client
import interaction_log
import vbs_audit_router
import diagnostics_router
import benchmark_router
BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent.parent
DATASETS_DIR = WORKSPACE_ROOT / "datasets"
LOG_FILE_PATH = BACKEND_DIR / "preprocessing.log"

# Add directories to sys.path: ensure BACKEND_DIR has highest precedence
sys.path.insert(0, str(BACKEND_DIR))
sys.path.append(str(WORKSPACE_ROOT))
sys.path.append(str(WORKSPACE_ROOT / "inference-code"))
app = FastAPI(title="Multimedia Retrieval API", version="1.0.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register VBS Audit, Diagnostics, and Benchmark routers
app.include_router(vbs_audit_router.router)
app.include_router(vbs_audit_router.sotuyen_router)
app.include_router(diagnostics_router.router)
app.include_router(benchmark_router.router)
from search.reranker import rerank_with_tail

def load_vlm():
    import config
    from models.qwen_vlm import QwenVLM
    from models.openai_vlm import OpenAIVLM
    if config.VLM_OPTION == "local":
        return QwenVLM()
    elif config.VLM_OPTION == "openai":
        return OpenAIVLM()
    else:
        raise ValueError(f"Unknown VLM option: {config.VLM_OPTION}")

def load_embedder():
    import config
    from models.embedding import WeMMEmbedding4BEmbedder, DashScopeCloudEmbedder
    if config.EMBEDDING_OPTION == "local":
        return WeMMEmbedding4BEmbedder()
    elif config.EMBEDDING_OPTION == "cloud":
        return DashScopeCloudEmbedder()
    else:
        raise ValueError(f"Unknown embedding option: {config.EMBEDDING_OPTION}")
def load_secondary_embedder():
    """None when disabled - see models/siglip_embedder.py."""
    import config
    if not config.SECONDARY_EMBEDDER_ENABLED:
        return None
    from models.siglip_embedder import SigLIPEmbedder
    return SigLIPEmbedder()

# 2. Services Management (Lazy Singletons)
_vlm = None
_detector = None
_embedder = None
_secondary_embedder = None
_query_proc = None
_searcher = None
_reranker = None
_services_lock = threading.Lock()
_preprocess_process = None
_preprocess_logs = []
# Single global interactive-session state, matching the "one operator per
# backend instance" model (no per-user session_id/store needed). "history"
# holds past {query, answer} turns for QueryProcessor.rewrite_query_cqr;
# "last_query_vector" holds the most recent dense query vector so
# /api/feedback has a base vector to Rocchio-adjust from.
_session_state = {
    "history": [],
    "last_query_vector": None,
    # KIS-C clarification round-trip: set when the previous turn asked a
    # clarifying question; holds that question plus the candidate ids it was
    # generated from, so the next turn's answer can boost exactly those
    # candidates (Sekulic et al. arXiv:2008.03717). Consumed (reset to None)
    # by every /api/search.
    "pending_clarification": None,
    # {point_id: {"source_file", "caption"}} for the last search's fused pool,
    # so /api/feedback can describe the operator's accepted/rejected picks in
    # words for the next CQR rewrite without a Qdrant payload fetch.
    "last_candidate_info": {},
}
# DRES session - single global value, matching the "one operator per
# backend instance" model (see plan) rather than a per-user session store.
_dres_session_id = None
# AVS duplicate-video submission guard: task_id -> set of video names
# already submitted for that task. Reset naturally per task_id (a fresh
# task_id starts with an empty set via .get(task_id, set())) - no explicit
# reset needed when DRES moves to the next task.
_avs_submitted_by_task: dict = {}
# KIS-C clarification-question trigger (see HybridSearcher.compute_ambiguity_score):
# 0.0-1.0 ratio of distinct videos among the top candidates; above this,
# /api/search generates a clarifying question instead of just returning
# results the operator has to disambiguate unaided. Webapp-only knob (not
# used by CLI/batch_query.py/evaluation), so it lives here rather than
# inference-code/config.py.
AMBIGUITY_THRESHOLD = float(os.getenv("AMBIGUITY_THRESHOLD", "0.7"))


def _check_avs_duplicate(task_id: str, video_name: Optional[str], force: bool, submitted_by_task: dict) -> Optional[dict]:
    """
    Pure-logic AVS duplicate-video guard (VBS_GUIDE.md §5.2: resubmitting an
    already-scored video earns no additional credit, and a wrong submission
    before the first correct one penalizes the whole video), factored out
    of dres_submit() so it's testable without importing fastapi/cv2/etc.
    Returns a warning dict if this submission should be blocked (soft - the
    caller can override with force=True), else None.
    """
    if not video_name or force:
        return None
    if video_name in submitted_by_task.get(task_id, set()):
        return {
            "warning": (
                f"Video '{video_name}' already submitted for task '{task_id}' - "
                "VBS's AVS scoring gives no additional credit for a second shot "
                "from an already-scored video. Resend with force=true to submit anyway."
            ),
            "video_name": video_name,
            "task_id": task_id,
        }
    return None

def _dres_config():
    """
    Reads DRES_BASE_URL/DRES_USERNAME/DRES_PASSWORD/DRES_EVALUATION_ID from
    the environment. `import config` first so inference-code/config.py's
    load_dotenv() side effect has run at least once in this process (this
    module has no separate .env loading of its own).
    """
    import config  # noqa: F401 - triggers load_dotenv() as a side effect
    return {
        "base_url": os.getenv("DRES_BASE_URL", ""),
        "username": os.getenv("DRES_USERNAME", ""),
        "password": os.getenv("DRES_PASSWORD", ""),
        "evaluation_id": os.getenv("DRES_EVALUATION_ID", ""),
    }

def init_services(query_type: int = 1):
    """
    Initialize query processing and search services dynamically.
    """
    global _vlm, _detector, _embedder, _secondary_embedder, _query_proc, _searcher, _reranker
    with _services_lock:
        # Check env variables inside function to allow updates
        import config
        from models.object_detector import ObjectDetector
        from search.query_processor import QueryProcessor
        from search.hybrid_search import HybridSearcher
        from search.reranker import Reranker

        current_model = getattr(_vlm, "model_name", None)
        target_model = os.getenv("OPENAI_VLM_MODEL_NAME")
        if _vlm is None or (current_model and target_model and current_model != target_model):
            print(f"Initializing VLM (switching from {current_model} to {target_model})...")
            _vlm = load_vlm()
            _query_proc = None
            _reranker = None
        if _embedder is None:
            print("Initializing Embedder...")
            _embedder = load_embedder()
            _secondary_embedder = load_secondary_embedder()

        if _detector is None and query_type == 2:
            print("Initializing Object Detector...")
            _detector = ObjectDetector(option=config.DETECTOR_OPTION)

        if _query_proc is None:
            _query_proc = QueryProcessor(vlm_client=_vlm)

        if _searcher is None:
            _searcher = HybridSearcher(embedder=_embedder, secondary_embedder=_secondary_embedder)

        # Re-initialize reranker if detector is newly loaded
        if _reranker is None or (_detector is not None and _reranker.detector is None):
            _reranker = Reranker(vlm_client=_vlm, detector_client=_detector)

        return _query_proc, _searcher, _reranker

# Ensure datasets directory exists
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_dataset_dir(requested: Optional[str]) -> str:
    """Resolve a dataset directory without allowing the API to escape it."""
    dataset_root = os.path.realpath(str(DATASETS_DIR))
    raw_path = requested or dataset_root
    candidate = raw_path if os.path.isabs(raw_path) else os.path.join(dataset_root, raw_path)
    resolved = os.path.realpath(candidate)
    try:
        inside_root = os.path.commonpath((dataset_root, resolved)) == dataset_root
    except ValueError:
        inside_root = False
    if not inside_root:
        raise HTTPException(status_code=400, detail="dataset_dir must be inside the server dataset directory")
    return resolved


def _resolve_media_path(video_name: str) -> Path:
    """Resolve a frontend media name under DATASETS_DIR and require a file."""
    if not isinstance(video_name, str) or not video_name.strip():
        raise HTTPException(status_code=400, detail="video_name is required")

    clean_name = video_name.strip()
    dataset_root = Path(os.path.realpath(str(DATASETS_DIR)))
    stem = Path(clean_name).stem

    candidates = [
        dataset_root / clean_name,
        dataset_root / "v3c" / "videos" / clean_name,
        dataset_root / "v3c" / "videos" / f"{stem}.mp4",
        dataset_root / "videos" / clean_name,
        dataset_root / "videos" / f"{stem}.mp4",
        dataset_root / "v3c" / "keyframes" / clean_name,
        dataset_root / "keyframes" / clean_name,
        dataset_root / "v3c-sample" / clean_name,
    ]

    for cand in candidates:
        try:
            resolved = cand.resolve()
            if resolved.is_file():
                if os.path.commonpath((str(dataset_root), str(resolved))) == str(dataset_root):
                    return resolved
        except Exception:
            continue

    raise HTTPException(status_code=404, detail=f"Media file '{video_name}' not found")

def _vqa_public_evidence(candidate: dict) -> dict:
    """Expose one validated, dataset-relative media reference to the UI."""
    media_name = None
    raw_path = candidate.get("vqa_evidence_path")
    if isinstance(raw_path, str) and raw_path:
        dataset_root = os.path.realpath(str(DATASETS_DIR))
        resolved = os.path.realpath(raw_path)
        try:
            if os.path.commonpath((dataset_root, resolved)) == dataset_root and os.path.isfile(resolved):
                media_name = os.path.relpath(resolved, dataset_root)
        except ValueError:
            media_name = None

    frame_idx = candidate.get("vqa_evidence_frame_idx")
    if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
        frame_idx = None
    timestamp = candidate.get("vqa_evidence_timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        timestamp = None
    elif not math.isfinite(float(timestamp)) or float(timestamp) < 0:
        timestamp = None
    else:
        timestamp = float(timestamp)
    return {
        "evidence_media_name": media_name,
        "evidence_frame_idx": frame_idx,
        "evidence_timestamp": timestamp,
    }


def _public_vqa_payload(payload: dict) -> dict:
    """Remove local paths and malformed temporal fields from VQA JSON."""
    public_payload = dict(payload or {})
    public_payload.pop("keyframe_path", None)
    public_payload.pop("frame_path", None)

    raw_frame_idx = public_payload.get("frame_idx")
    if isinstance(raw_frame_idx, bool):
        public_payload.pop("frame_idx", None)
    else:
        try:
            normalized_frame_idx = float(raw_frame_idx)
            if not math.isfinite(normalized_frame_idx) or normalized_frame_idx < 0 or not normalized_frame_idx.is_integer():
                raise ValueError
            public_payload["frame_idx"] = int(normalized_frame_idx)
        except (TypeError, ValueError):
            public_payload.pop("frame_idx", None)

    raw_timestamp = public_payload.get("timestamp")
    if isinstance(raw_timestamp, bool):
        public_payload.pop("timestamp", None)
    else:
        try:
            normalized_timestamp = float(raw_timestamp)
            if not math.isfinite(normalized_timestamp) or normalized_timestamp < 0:
                raise ValueError
            public_payload["timestamp"] = normalized_timestamp
        except (TypeError, ValueError):
            public_payload.pop("timestamp", None)
    return public_payload

# 3. Request Models
class SearchRequest(BaseModel):
    type: int
    query: str
    dataset_dir: Optional[str] = None
    # Escalate-precision-on-demand (U-Cker/PraK-inspired, VBS2026): when
    # None (default, unchecked in the UI), server config defaults
    # (QDRANT_EXACT_SEARCH / VERIFICATION_RERANK_ENABLED) apply unchanged.
    # Set explicitly to override for just this search, e.g. when an
    # operator is stuck on a hard KIS-T task and wants to trade latency for
    # precision without editing .env/restarting the backend.
    exact: Optional[bool] = None
    verify: Optional[bool] = None
    # Graduated middle ground between exact=False (fast/default) and
    # exact=True (full brute-force) - raises Qdrant's HNSW search-time
    # candidate-list size for higher recall at a smaller latency cost than
    # exact search. None (default) leaves Qdrant's own index default in effect.
    hnsw_ef: Optional[int] = None
    # KIS-C: the operator's answer to the clarifying question asked last
    # turn, sent separately from `query` (which still carries it appended,
    # for retrieval) so the backend can boost the exact candidates the
    # question was about without string-parsing the composed query.
    clarification_answer: Optional[str] = None

class FeedbackRequest(BaseModel):
    positive_ids: List[str] = []
    negative_ids: List[str] = []
    top_k: int = 20

class QueryByExampleRequest(BaseModel):
    point_id: str
    top_k: int = 20

class TemporalSearchRequest(BaseModel):
    queries: List[str]
    window_frames: int = 150
    top_k: int = 15

class DresSubmitRequest(BaseModel):
    task_id: str
    payload: dict
    # AVS duplicate-video guard (VBS_GUIDE.md §5.2: resubmitting an
    # already-scored video earns no additional credit, and each wrong
    # submission before the first correct one penalizes the whole video) -
    # optional so KIS/VQA submissions (which don't need this check) can
    # simply omit video_name.
    video_name: Optional[str] = None
    force: bool = False

class InVideoSearchRequest(BaseModel):
    query: str
    video_name: str
    dataset_dir: Optional[str] = None

class InVideoRerankRequest(BaseModel):
    query: str
    top_k: int = 20
# 4. Helper to Get DB Stats
def get_qdrant_stats():
    try:
        from qdrant_client import QdrantClient
        import config
        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY if config.QDRANT_API_KEY else None
        )
        collections_list = []
        total_points = 0
        
        cols_res = client.get_collections()
        for c in cols_res.collections:
            try:
                info = client.get_collection(collection_name=c.name)
                vectors_cfg = getattr(info.config.params, "vectors", None)
                dim = getattr(vectors_cfg, "size", 2048) if vectors_cfg else 2048
                dist = str(getattr(vectors_cfg, "distance", "Cosine")) if vectors_cfg else "Cosine"
                points = getattr(info, "points_count", 0) or 0
                indexed = getattr(info, "indexed_vectors_count", 0) or 0
                total_points += points
                collections_list.append({
                    "name": c.name,
                    "points": points,
                    "indexed": indexed,
                    "dim": dim,
                    "distance": dist,
                    "status": str(getattr(info, "status", "green")),
                })
            except Exception as sub_err:
                collections_list.append({
                    "name": c.name,
                    "points": 0,
                    "indexed": 0,
                    "dim": 2048,
                    "distance": "Cosine",
                    "status": "error",
                    "error": str(sub_err)
                })

        visual_col = next((c for c in collections_list if "visual" in c["name"]), None)
        audio_col = next((c for c in collections_list if "audio" in c["name"]), None)
        
        return {
            "status": "connected",
            "host": config.QDRANT_HOST,
            "port": config.QDRANT_PORT,
            "collections": collections_list,
            "total_points": total_points,
            "visual_points": visual_col["points"] if visual_col else 0,
            "audio_points": audio_col["points"] if audio_col else 0,
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e)
        }

# 5. API Endpoints

@app.get("/api/status")
def get_status():
    """
    Get backend status including models configuration, DB stats, and scanned datasets.
    """
    # Load configuration values
    import config
    
    # List files in datasets folder
    supported_exts = ('.mp4', '.avi', '.mkv', '.mov', '.jpg', '.jpeg', '.png', '.mp3', '.wav', '.m4a')
    files_list = []
    if DATASETS_DIR.exists():
        for item in DATASETS_DIR.iterdir():
            if item.is_file() and item.suffix.lower() in supported_exts:
                files_list.append({
                    "name": item.name,
                    "size_mb": round(item.stat().st_size / (1024 * 1024), 2),
                    "type": "video" if item.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov') else 
                            "image" if item.suffix.lower() in ('.jpg', '.jpeg', '.png') else "audio"
                })

    db_stats = get_qdrant_stats()
    
    # Check if preprocessing is running
    is_preprocessing = _preprocess_process is not None and _preprocess_process.poll() is None

    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "vlm_option": config.VLM_OPTION,
        "detector_option": config.DETECTOR_OPTION,
        "qdrant": db_stats,
        "dataset_files": files_list,
        "preprocessing_active": is_preprocessing
    }

@app.post("/api/search")
async def run_search(request: SearchRequest):
    """
    Execute Type 1 (Textual-KIS), Type 2 (VQA), Type 3 (Temporal-Alignment), or Type 4 (AVS) search.
    """
    if request.type not in [1, 2, 3, 4]:
        raise HTTPException(status_code=400, detail="Invalid search type. Must be 1, 2, 3, or 4.")

    dataset_dir = _resolve_dataset_dir(request.dataset_dir)
    try:
        # Initialize services dynamically
        query_proc, searcher, reranker = init_services(query_type=request.type)
        
        # 0. CQR: resolve pronouns/implicit references against this
        # session's prior turns (QueryProcessor.rewrite_query_cqr existed
        # since the AIC-era code but was never called anywhere - VBS's
        # multi-turn interactive session is exactly what it was written
        # for). No-op (returns the query unchanged) on the first search of
        # a session, when history is empty.
        resolved_query = query_proc.rewrite_query_cqr(request.query, _session_state["history"])

        # 1. Query Processing
        hyde_query = query_proc.generate_hyde(resolved_query)

        # 2. Candidate Retrieval
        query_hits = searcher.search(resolved_query, top_k=15, exact=request.exact, hnsw_ef=request.hnsw_ef)
        hyde_hits = searcher.search(hyde_query, top_k=15, exact=request.exact, hnsw_ef=request.hnsw_ef)
        secondary_hits = searcher.dense_search_secondary(
            resolved_query, top_k=15, exact=request.exact, hnsw_ef=request.hnsw_ef
        )
        # labels=[...] (VIREO/SnapMind/NII-UIT-inspired explainability,
        # VBS2026): tags each fused hit with which source(s) it came from
        # ("query" = original text, "hyde" = HyDE hypothetical description,
        # "secondary" = the secondary embedder ensemble) so the operator can
        # see WHY a result matched instead of one opaque combined score.
        candidates = searcher.merge_rrf(
            query_hits, hyde_hits, secondary_hits, labels=["query", "hyde", "secondary"]
        )
        # TAG-inspired (arXiv:2508.07925) temporal coherence re-scoring:
        # boost candidates that have other same-video candidates nearby in
        # frame_idx, so a real event isn't left fragmented across several
        # marginal individual scores. Run right after merge_rrf, before the
        # type-specific reranking below.
        candidates = searcher.temporal_coherence_boost(candidates)

        # Result Diversification (Khoa: Adaptive Sampling & Retrieval
        # Accuracy) - collapses to the single highest-scoring hit per
        # (video, scene) so the pool isn't flooded by several near-duplicate
        # keyframes of the same event. CLI/batch_query.py/evaluation already
        # call this after their own merge_rrf; the webapp flow now does too,
        # directly serving AVS's diversity-across-videos scoring
        # (VBS_GUIDE.md §4.2/§5.2) as well as giving KIS-T/KIS-C/VQA
        # operators a more varied result set to scan.
        import config
        candidates = searcher.diversify_by_scene(candidates, top_k=config.SUBMISSION_TOP_K)

        # Remember this turn's resolved query + dense vector for later
        # session actions: /api/feedback Rocchio-adjusts from
        # last_query_vector, and history lets the NEXT search's CQR
        # rewrite resolve references back to this one.
        _session_state["last_query_vector"] = searcher.embedder.embed_text(resolved_query)
        _session_state["history"].append({"query": resolved_query})

        # Cache {id: {source_file, caption}} for this turn's fused pool so
        # /api/feedback can describe the operator's accepted/rejected picks
        # in words for the next CQR rewrite, without an extra Qdrant payload
        # fetch. Rebuilt (not appended) every search - naturally bounded.
        _session_state["last_candidate_info"] = {
            c["id"]: {
                "source_file": (c.get("payload") or {}).get("source_file"),
                "caption": ((c.get("payload") or {}).get("caption") or "")[:200],
            }
            for c in candidates
        }

        # Consume the KIS-C clarification flag once per search, regardless of
        # query type, so a stale flag can never boost a much later unrelated
        # turn (see kis_c_scoring.boost_by_clarification_answer, applied only
        # in the Type 1 branch below).
        pending_clarification = _session_state["pending_clarification"]
        _session_state["pending_clarification"] = None
        clarification_boost_applied = False

        if not candidates:
            return {
                "query": request.query,
                "type": request.type,
                "results": [],
                "message": "No candidate frames retrieved from database."
            }

        # 3. Type-specific Reranking
        results = []
        clarification_question = None
        if request.type == 1:
            # Type 1: Textual-KIS
            import config
            from search.kis_c_scoring import boost_by_clarification_answer, apply_conversational_negative_filter

            # KIS-C clarification-answer boost (Sekulic et al. arXiv:2008.03717):
            if pending_clarification and (request.clarification_answer or "").strip():
                candidates = boost_by_clarification_answer(
                    candidates,
                    pending_clarification.get("candidate_ids") or [],
                    request.clarification_answer,
                )
                clarification_boost_applied = True

            # Conversational negative feedback filtering (Rocchio/Exquisitor loop):
            if _session_state.get("history"):
                recent_rejected = []
                for turn in _session_state["history"][-2:]:
                    recent_rejected.extend(turn.get("rejected", []))
                if recent_rejected:
                    candidates = apply_conversational_negative_filter(candidates, recent_rejected)
            # KIS-C clarification (CAR/ambiguity-detection-inspired): if the
            # fused candidate pool is spread across many unrelated videos
            # with no clear winner, ask the operator a clarifying question
            # instead of silently returning an under-specified result set.
            # Gated behind AMBIGUITY_THRESHOLD so the common (unambiguous)
            # case pays zero extra VLM-call cost.
            ambiguity_score = searcher.compute_ambiguity_score(candidates)
            if ambiguity_score >= AMBIGUITY_THRESHOLD:
                seen_videos = set()
                summaries = []
                summary_ids = []
                for c in candidates:
                    video = c["payload"].get("source_file")
                    if video in seen_videos:
                        continue
                    seen_videos.add(video)
                    summaries.append(c["payload"].get("caption") or video or "")
                    summary_ids.append(c["id"])
                    if len(summaries) >= 5:
                        break
                clarification_question = query_proc.generate_clarification_question(resolved_query, summaries)
                if clarification_question:
                    _session_state["pending_clarification"] = {
                        "question": clarification_question,
                        "candidate_ids": summary_ids,
                    }

            rerank_k = getattr(config, "RERANK_TOP_K", 20)
            submission_k = getattr(config, "SUBMISSION_TOP_K", 100)
            top_candidates = rerank_with_tail(
                lambda c: reranker.rerank_type1(resolved_query, c, verify=request.verify),
                candidates, rerank_k, submission_k,
            )
            # Threshold only candidates that actually went through the VLM
            # rerank. rerank_with_tail's un-reranked tail carries rrf_score
            # (~0.02-0.1, RRF k=60 scale), which can never clear a rerank-score
            # threshold - the old fallback treated it as 0.0 and silently
            # emptied the deep pool rerank_with_tail preserves.
            top_candidates = [
                c for c in top_candidates
                if "rerank_score" not in c or c["rerank_score"] >= config.RERANK_SCORE_THRESHOLD
            ]
            for idx, c in enumerate(top_candidates):
                results.append({
                    "rank": idx + 1,
                    "score": c.get("rerank_score", c.get("rrf_score", 0.0)),
                    "id": c["id"],
                    "payload": c["payload"],
                    "matched_via": c.get("matched_via", [])
                })
                
        elif request.type == 2:
            # Type 2: VQA
            decomp = query_proc.decompose_query(resolved_query)
            sub_queries = decomp.get("sub_queries", [resolved_query])

            # In-Video Retrieval used to run automatically here on every
            # search (see HybridSearcher.in_video_refine) - fine for AIC's
            # 4-hour batch window, too slow as a default per-query cost for
            # VBS's live 5-7 minute task clock. Now a manual action the
            # operator triggers explicitly via /api/in-video-search once
            # they've spotted a promising video in the initial results.
            rerank_k = getattr(config, "RERANK_TOP_K", 20)
            submission_k = getattr(config, "SUBMISSION_TOP_K", 100)
            top_candidates = rerank_with_tail(
                lambda c: reranker.rerank_type2_vqa(
                    resolved_query, sub_queries, c, dataset_dir, verify=request.verify
                ),
                candidates, rerank_k, submission_k,
            )
            
            for idx, c in enumerate(top_candidates):
                # The reranker already answered this exact candidate/frame.
                # Do not issue a second ungrounded call here: that used to
                # answer from a possibly missing frame and could leak a
                # result that did not correspond to the ranked candidate.
                is_answer_candidate = idx == 0
                answer = c.get("vqa_answer", "UNKNOWN") if is_answer_candidate else None
                vqa_answer_valid = is_answer_candidate and c.get("vqa_answer_valid", False)
                if is_answer_candidate and not vqa_answer_valid:
                    answer = "N/A"

                results.append({
                    "rank": idx + 1,
                    "score": c.get("final_score", 0.0),
                    "vqa_score": c.get("vqa_score", 0.0),
                    "rrf_score": c.get("rrf_score", 0.0),
                    "id": c["id"],
                    "payload": _public_vqa_payload(c["payload"]),
                    "answer": answer,
                    "vqa_answer": c.get("vqa_answer", "UNKNOWN") if is_answer_candidate else None,
                    "vqa_answer_valid": vqa_answer_valid,
                    "vqa_evidence_available": c.get("vqa_evidence_available", False),
                    "vqa_evidence_reason": c.get("vqa_evidence_reason", ""),
                    "answer_candidate_id": c.get("vqa_candidate_id") if vqa_answer_valid else None,
                    "answer_video_id": c.get("vqa_video_id") if vqa_answer_valid else None,
                    "answer_frame_idx": c.get("vqa_frame_idx") if vqa_answer_valid else None,
                    **_vqa_public_evidence(c),
                    "matched_via": c.get("matched_via", [])
                })
            # Record the generated answer against this turn so a later CQR
            # rewrite (e.g. "was there a sign in that scene too?") can
            # resolve against what the system actually answered, not just
            # the query text.
            if results and results[0].get("vqa_answer_valid") and results[0].get("answer"):
                _session_state["history"][-1]["answer"] = results[0]["answer"]

        elif request.type == 3:
            # Type 3: Temporal Alignment
            top_sequences = reranker.rerank_type3_temporal(resolved_query, candidates[:20], query_proc, searcher)
            for idx, seq in enumerate(top_sequences):
                # Format to look like candidate output but grouped
                results.append({
                    "rank": idx + 1,
                    "score": seq["score"],
                    "video_name": seq["video_name"],
                    "timestamps": seq["timestamps"],
                    "frame_ids": seq["frame_ids"],
                    "payload": {
                        "source_file": seq["video_name"],
                        "timestamp": seq["timestamps"][0] if seq["timestamps"] else 0.0,
                        "caption": f"Sequence of {len(seq['frame_ids'])} frames. Timestamps: {seq['timestamps']}"
                    }
                })
                
        elif request.type == 4:
            # Type 4: AVS (Ad-hoc Video Search) - Maximize cross-video and cross-scene diversity
            diverse_candidates = searcher.diversify_by_scene(candidates, top_k=getattr(config, "SUBMISSION_TOP_K", 100))
            for idx, c in enumerate(diverse_candidates):
                results.append({
                    "rank": idx + 1,
                    "score": c.get("score", c.get("rrf_score", 0.0)),
                    "id": c["id"],
                    "payload": c["payload"],
                    "matched_via": c.get("matched_via", [])
                })
        interaction_log.log_query(
            "search", resolved_query, [r.get("id") for r in results],
            dres_config=_dres_config(), session_id=_dres_session_id,
        )
        return {
            "query": request.query,
            "type": request.type,
            "results": results,
            "clarification": clarification_question,
            # Explainability (mirrors `matched_via`): whether this turn's
            # ranking was adjusted by kis_c_scoring.boost_by_clarification_answer.
            "clarification_boost_applied": clarification_boost_applied,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/feedback")
def run_feedback(request: FeedbackRequest):
    """
    Relevance feedback: Rocchio-adjusts the session's last query vector
    toward candidates the operator marked positive and away from ones
    marked negative (HybridSearcher.rocchio_adjust), then re-searches with
    that adjusted vector directly (dense_search_by_vector) - no new text
    query needed. Requires a prior /api/search call in this session to
    have set _session_state["last_query_vector"].
    """
    if _session_state["last_query_vector"] is None:
        raise HTTPException(status_code=400, detail="No active query in this session - run /api/search first")
    if not request.positive_ids and not request.negative_ids:
        raise HTTPException(status_code=400, detail="Provide at least one positive_ids or negative_ids")

    try:
        _, searcher, _ = init_services(query_type=1)
        positive_vectors = [v for pid in request.positive_ids if (v := searcher.get_point_vector(pid)) is not None]
        negative_vectors = [v for pid in request.negative_ids if (v := searcher.get_point_vector(pid)) is not None]

        adjusted_vector = searcher.rocchio_adjust(
            _session_state["last_query_vector"], positive_vectors, negative_vectors
        )
        _session_state["last_query_vector"] = adjusted_vector

        # Exquisitor-inspired (VBS 2024/2025 unified conversational +
        # relevance-feedback loop, lightest prompt-only form): the
        # operator's accept/reject signal was previously isolated in the
        # Rocchio vector and never reached CQR. Recording it on the current
        # history turn means the NEXT rewrite_query_cqr prompt knows which
        # readings were already rejected/confirmed. Same single CQR call as
        # before - prompt context only, no new call.
        from search.conversational_context import record_feedback_in_history
        record_feedback_in_history(
            _session_state["history"], _session_state["last_candidate_info"],
            request.positive_ids, request.negative_ids,
        )

        hits = searcher.dense_search_by_vector(adjusted_vector, top_k=request.top_k)
        results = [{"rank": idx + 1, "score": hit["score"], "id": hit["id"], "payload": hit["payload"]}
                   for idx, hit in enumerate(hits)]
        interaction_log.log_query(
            "feedback",
            f"positive={request.positive_ids} negative={request.negative_ids}",
            [r["id"] for r in results],
            dres_config=_dres_config(), session_id=_dres_session_id,
        )
        return {"results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feedback search failed: {str(e)}")

@app.post("/api/query-by-example")
def run_query_by_example(request: QueryByExampleRequest):
    """
    Query-by-example: reuses an already-indexed point's own stored dense
    vector (HybridSearcher.get_point_vector) as the next query, instead of
    re-embedding an image - the operator clicks a result and searches for
    "more like this" directly.
    """
    try:
        _, searcher, _ = init_services(query_type=1)
        vector = searcher.get_point_vector(request.point_id)
        if vector is None:
            raise HTTPException(status_code=404, detail=f"No stored vector found for point '{request.point_id}'")

        _session_state["last_query_vector"] = vector
        hits = searcher.dense_search_by_vector(vector, top_k=request.top_k)
        results = [{"rank": idx + 1, "score": hit["score"], "id": hit["id"], "payload": hit["payload"]}
                   for idx, hit in enumerate(hits)]
        interaction_log.log_query(
            "query_by_example", f"point_id={request.point_id}", [r["id"] for r in results],
            dres_config=_dres_config(), session_id=_dres_session_id,
        )
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query-by-example failed: {str(e)}")

@app.post("/api/search-by-image")
async def run_search_by_image(file: UploadFile = File(...), top_k: int = Form(20)):
    """
    KIS-V (Visual KIS, VBS_GUIDE.md §4.1): the operator is shown a short
    clip on the projector - not a text description - so the natural query
    is "search by what I just saw", not text. Embeds an uploaded photo/
    screenshot of the target moment with the same visual embedder used at
    indexing time and searches directly by that vector. No RRF fusion
    here (unlike /api/search) - there's only one ranked list, no text/
    HyDE/secondary-embedder branches to fuse an uploaded image against.
    """
    try:
        _, searcher, _ = init_services(query_type=1)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        vector = searcher.embedder.embed_image(image)

        _session_state["last_query_vector"] = vector
        hits = searcher.dense_search_by_vector(vector, top_k=top_k)
        results = [{"rank": idx + 1, "score": hit["score"], "id": hit["id"], "payload": hit["payload"]}
                   for idx, hit in enumerate(hits)]
        interaction_log.log_query(
            "search_by_image", f"uploaded_image:{file.filename}", [r["id"] for r in results],
            dres_config=_dres_config(), session_id=_dres_session_id,
        )
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search-by-image failed: {str(e)}")

@app.post("/api/search-by-video")
async def run_search_by_video(file: UploadFile = File(...), top_k: int = Form(20)):
    """
    KIS-V clip search: sample representative frames from the uploaded visual
    prompt, embed each frame, and merge the strongest indexed matches. The
    upload is temporary and is never added to the dataset or served back.
    """
    temp_path = None
    try:
        suffix = Path(file.filename or "prompt.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_count <= 0 or fps <= 0:
            cap.release()
            raise HTTPException(status_code=400, detail="Unable to read frames from the uploaded video")

        sample_count = min(8, frame_count)
        frame_positions = [
            int(index * (frame_count - 1) / max(sample_count - 1, 1))
            for index in range(sample_count)
        ]
        sampled_frames = []
        for frame_idx in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if success:
                sampled_frames.append((frame_idx / fps, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        cap.release()

        if not sampled_frames:
            raise HTTPException(status_code=400, detail="No readable frames found in the uploaded video")

        _, searcher, _ = init_services(query_type=1)
        merged_hits = {}
        last_visual_vector = None
        per_frame_top_k = max(5, min(int(top_k), 20))
        for clip_timestamp, frame_image in sampled_frames:
            vector = searcher.embedder.embed_image(frame_image)
            last_visual_vector = vector
            for hit in searcher.dense_search_by_vector(vector, top_k=per_frame_top_k):
                existing = merged_hits.get(hit["id"])
                if existing is None or hit["score"] > existing["score"]:
                    merged_hits[hit["id"]] = {
                        **hit,
                        "matched_clip_timestamp": round(clip_timestamp, 3),
                    }

        ranked_hits = sorted(merged_hits.values(), key=lambda hit: hit["score"], reverse=True)[:int(top_k)]
        results = [
            {
                "rank": index + 1,
                "score": hit["score"],
                "id": hit["id"],
                "payload": hit["payload"],
                "matched_clip_timestamp": hit["matched_clip_timestamp"],
            }
            for index, hit in enumerate(ranked_hits)
        ]
        _session_state["last_query_vector"] = last_visual_vector
        interaction_log.log_query(
            "search_by_video",
            f"uploaded_video:{file.filename}",
            [result["id"] for result in results],
            dres_config=_dres_config(),
            session_id=_dres_session_id,
        )
        return {"results": results, "sampled_frames": len(sampled_frames)}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search-by-video failed: {str(e)}")
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

@app.post("/api/temporal-search")
def run_temporal_search(request: TemporalSearchRequest):
    """
    VBS-style temporal chain query: N>=2 sequential text descriptions ("a
    bicycle passes, then a red car, then a dog runs by") searched
    independently, then combined via HybridSearcher.temporal_chain_match
    into per-video frame chains where each step's frame occurs after the
    previous step's within window_frames.
    """
    if len(request.queries) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 queries for a temporal chain")
    try:
        _, searcher, _ = init_services(query_type=1)
        hit_lists = [searcher.search(q, top_k=request.top_k) for q in request.queries]
        matches = searcher.temporal_chain_match(hit_lists, window_frames=request.window_frames)
        results = [
            {
                "rank": idx + 1,
                "score": m["score"],
                "video_name": m["video_name"],
                "frames": m["frames"],
                "payloads": m["payloads"],
            }
            for idx, m in enumerate(matches)
        ]
        interaction_log.log_query(
            "temporal_search", " -> ".join(request.queries),
            [r["video_name"] for r in results],
            dres_config=_dres_config(), session_id=_dres_session_id,
        )
        return {"queries": request.queries, "results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Temporal search failed: {str(e)}")

@app.get("/api/browse-video/{video_name}")
def browse_video(video_name: str):
    """
    Full keyframe browse for one specific video - reuses
    HybridSearcher.get_all_points_for_video() verbatim (already built for
    TRAKE's DP alignment), sorted by frame_idx, so the operator can page
    through everything indexed for a video they've spotted as promising.
    """
    try:
        _, searcher, _ = init_services(query_type=1)
        points = searcher.get_all_points_for_video(video_name)
        points = [p for p in points if p["payload"].get("frame_idx") is not None]
        points.sort(key=lambda p: p["payload"]["frame_idx"])
        return {
            "video_name": video_name,
            "frames": [{"id": p["id"], "payload": p["payload"]} for p in points],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Browse video failed: {str(e)}")
@app.get("/api/video/{video_name}/timeline")
def get_video_timeline(video_name: str, center_timestamp: Optional[float] = None, window_sec: float = 30.0):
    """
    In-Video Timeline Explorer (PraK V4 & NII-UIT inspired, VBS2026):
    Returns all indexed frames for the specified video, sorted chronologically,
    with an optional focus window around center_timestamp (+/- window_sec).
    """
    try:
        _, searcher, _ = init_services(query_type=1)
        points = searcher.get_all_points_for_video(video_name)
        frames = []
        for p in points:
            payload = p.get("payload", {})
            frame_idx = payload.get("frame_idx")
            timestamp = payload.get("timestamp")
            if timestamp is None and payload.get("pts_time") is not None:
                timestamp = payload.get("pts_time")

            if center_timestamp is not None and timestamp is not None:
                if abs(timestamp - center_timestamp) > window_sec:
                    continue

            frames.append({
                "id": p["id"],
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "caption": payload.get("caption", ""),
                "ocr_text": payload.get("ocr_text", ""),
                "scene_narrative": payload.get("scene_narrative", ""),
                "payload": payload,
            })

        frames.sort(key=lambda f: (f["frame_idx"] if f["frame_idx"] is not None else 0, f["timestamp"] if f["timestamp"] is not None else 0.0))
        return {
            "video_name": video_name,
            "total_frames": len(frames),
            "frames": frames,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Timeline lookup failed: {str(e)}")

@app.post("/api/video/{video_name}/rerank")
def rerank_in_video(video_name: str, request: InVideoRerankRequest):
    """
    Sub-shot & In-Video Reranker (PraK V4 / Exquisitor inspired):
    Scores all indexed frames inside this specific video against a target sub-query
    using dense similarity + text token matching.
    """
    try:
        import numpy as np
        _, searcher, _ = init_services(query_type=1)
        points = searcher.get_all_points_for_video(video_name)
        if not points:
            return {"video_name": video_name, "query": request.query, "results": []}

        query_vector = searcher.embedder.embed_text(request.query)
        scored_frames = []
        for p in points:
            point_vec = searcher.get_point_vector(p["id"])
            dense_sim = 0.0
            if point_vec is not None:
                norm_q = np.linalg.norm(query_vector)
                norm_p = np.linalg.norm(point_vec)
                if norm_q > 0 and norm_p > 0:
                    dense_sim = float(np.dot(query_vector, point_vec) / (norm_q * norm_p))

            payload = p.get("payload", {})
            text_corpus = f"{payload.get('caption', '')} {payload.get('ocr_text', '')} {payload.get('scene_narrative', '')}".lower()
            text_sim = 0.0
            tokens = [t.lower() for t in request.query.split() if len(t) > 2]
            if tokens:
                matched_tokens = sum(1 for t in tokens if t in text_corpus)
                text_sim = matched_tokens / len(tokens)

            combined_score = 0.7 * dense_sim + 0.3 * text_sim
            scored_frames.append({
                "id": p["id"],
                "score": round(combined_score, 4),
                "dense_score": round(dense_sim, 4),
                "text_score": round(text_sim, 4),
                "frame_idx": payload.get("frame_idx"),
                "timestamp": payload.get("timestamp", payload.get("pts_time")),
                "payload": payload,
            })

        scored_frames.sort(key=lambda x: x["score"], reverse=True)
        return {
            "video_name": video_name,
            "query": request.query,
            "results": scored_frames[:request.top_k]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"In-video reranking failed: {str(e)}")

@app.post("/api/in-video-search")
async def run_in_video_search(request: InVideoSearchRequest):
    """
    In-Video Retrieval (NII-UIT-inspired, VBS2026), as a manual operator
    action instead of an automatic step in /api/search's Type 2 flow (see
    the comment removed from there). Call this once the operator has
    spotted a promising video in the initial results and wants to search
    its full frame timeline directly (HybridSearcher.in_video_refine) rather
    than only whatever made the original candidate pool.
    """
    dataset_dir = _resolve_dataset_dir(request.dataset_dir)
    try:
        query_proc, searcher, reranker = init_services(query_type=2)

        # Seed in_video_refine with a single synthetic candidate for the
        # requested video (top_videos=1 scopes it to exactly that video,
        # rather than re-deriving "top videos" from a full search pool).
        seed = [{"id": "seed", "rrf_score": 1.0, "payload": {"source_file": request.video_name}}]
        candidates = searcher.in_video_refine(request.query, seed, top_videos=1, top_frames_per_video=20)
        # Drop the synthetic seed placeholder (no real timestamp/payload) -
        # if it's the only thing left, the video has no indexed frames at all.
        candidates = [c for c in candidates if c["id"] != "seed"]
        if not candidates:
            return {"query": request.query, "video_name": request.video_name, "results": [],
                    "message": f"No indexed frames found for video '{request.video_name}'."}

        decomp = query_proc.decompose_query(request.query)
        sub_queries = decomp.get("sub_queries", [request.query])
        top_candidates = reranker.rerank_type2_vqa(request.query, sub_queries, candidates[:10], dataset_dir)

        results = []
        for idx, c in enumerate(top_candidates):
            is_answer_candidate = idx == 0
            answer = c.get("vqa_answer", "UNKNOWN") if is_answer_candidate else None
            vqa_answer_valid = is_answer_candidate and c.get("vqa_answer_valid", False)
            if is_answer_candidate and not vqa_answer_valid:
                answer = "N/A"
            results.append({
                "rank": idx + 1,
                "score": c.get("final_score", 0.0),
                "vqa_score": c.get("vqa_score", 0.0),
                "rrf_score": c.get("rrf_score", 0.0),
                "id": c["id"],
                "payload": _public_vqa_payload(c["payload"]),
                "answer": answer,
                "vqa_answer": c.get("vqa_answer", "UNKNOWN") if is_answer_candidate else None,
                "vqa_answer_valid": vqa_answer_valid,
                "vqa_evidence_available": c.get("vqa_evidence_available", False),
                "vqa_evidence_reason": c.get("vqa_evidence_reason", ""),
                "answer_candidate_id": c.get("vqa_candidate_id") if vqa_answer_valid else None,
                "answer_video_id": c.get("vqa_video_id") if vqa_answer_valid else None,
                "answer_frame_idx": c.get("vqa_frame_idx") if vqa_answer_valid else None,
                **_vqa_public_evidence(c),
            })
        return {"query": request.query, "video_name": request.video_name, "results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"In-video search failed: {str(e)}")

@app.get("/api/media/frame")
def get_frame(video_name: str, timestamp: Optional[float] = None, frame_idx: Optional[int] = None):
    """
    Extract or directly serve a keyframe image from disk or video.
    """
    if not isinstance(video_name, str) or not video_name.strip():
        raise HTTPException(status_code=400, detail="video_name is required")

    clean_name = video_name.strip()
    stem = Path(clean_name).stem
    dataset_root = Path(os.path.realpath(str(DATASETS_DIR)))

    # 1. Fast path: check if pre-extracted keyframe image already exists on disk
    if frame_idx is not None:
        keyframe_candidates = [
            dataset_root / "v3c" / "keyframes" / stem / f"{frame_idx}.jpg",
            dataset_root / "keyframes" / stem / f"{frame_idx}.jpg",
            dataset_root / "v3c" / "keyframes" / stem / f"{frame_idx}.png",
            dataset_root / "v3c-sample" / "official" / "keyframes" / stem / f"shot{stem}_1_RKF.png",
        ]
        for kp in keyframe_candidates:
            if kp.is_file():
                return FileResponse(str(kp), media_type="image/jpeg" if kp.suffix == ".jpg" else "image/png")

    # If only timestamp is provided, check if close keyframe exists in keyframes dir
    if timestamp is not None and frame_idx is None:
        keyframe_dir = dataset_root / "v3c" / "keyframes" / stem
        if keyframe_dir.is_dir():
            est_idx = int(timestamp * 25.0)
            direct_file = keyframe_dir / f"{est_idx}.jpg"
            if direct_file.is_file():
                return FileResponse(str(direct_file), media_type="image/jpeg")
            frame_files = list(keyframe_dir.glob("*.jpg"))
            if frame_files:
                closest = min(frame_files, key=lambda f: abs(int(f.stem) - est_idx) if f.stem.isdigit() else 999999)
                if closest.is_file() and abs(int(closest.stem) - est_idx) <= 75:
                    return FileResponse(str(closest), media_type="image/jpeg")

    # 2. Slow path / fallback: resolve video file and decode frame with cv2
    video_path = _resolve_media_path(video_name)

    if str(video_path).lower().endswith(('.jpg', '.jpeg', '.png')):
        return FileResponse(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    if frame_idx is not None:
        canonical_frame_idx = max(0, frame_idx)
    elif timestamp is not None and math.isfinite(timestamp):
        canonical_frame_idx = max(0, int(timestamp * fps))
    else:
        canonical_frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, canonical_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=404, detail="Unable to decode requested frame from video")

    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode frame to JPEG")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/api/media/video/{video_name}")
def get_video(video_name: str):
    """
    Serve video file directly with support for seeking/range queries.
    """
    video_path = _resolve_media_path(video_name)
    return FileResponse(str(video_path))

@app.get("/api/media/{file_path:path}")
def get_media_file(file_path: str):
    """
    Serve a media file from the dataset directory by its dataset-relative
    path, or by bare video/image id (resolved through the audit router's
    recursive lookup). Backs the audit workspaces' candidate thumbnails,
    which reference /api/media/{src_file} and /api/media/{video_id}.mp4.
    Must stay registered after /api/media/frame and /api/media/video.
    """
    if not file_path.strip():
        raise HTTPException(status_code=400, detail="media file path is required")
    if "/" not in file_path and "\\" not in file_path and file_path.lower().endswith((".mp4", ".jpg", ".jpeg", ".png")):
        # Bare id: the file usually lives in a nested dataset subfolder, so
        # resolve it the same way vbs_audit_router builds thumbnail URLs.
        stem = file_path.rsplit(".", 1)[0]
        src_rel = vbs_audit_router.resolve_video_src_file(stem)
        media_path = _resolve_media_path(src_rel)
    else:
        media_path = _resolve_media_path(file_path)
    return FileResponse(str(media_path))

# 5b. DRES Integration (VBS_GUIDE.md section 6) - proxied through this
# backend so DRES_USERNAME/DRES_PASSWORD never reach the frontend.

@app.post("/api/dres/login")
def dres_login():
    """
    Logs into DRES using DRES_BASE_URL/DRES_USERNAME/DRES_PASSWORD from the
    environment, stores the session id in-memory (single global session -
    see _dres_session_id above). Call this once at the start of an
    operating session; re-call if a later DRES call reports the session
    expired.
    """
    global _dres_session_id
    cfg = _dres_config()
    if not cfg["base_url"] or not cfg["username"] or not cfg["password"]:
        raise HTTPException(status_code=400, detail="DRES_BASE_URL/DRES_USERNAME/DRES_PASSWORD not configured")
    try:
        _dres_session_id = dres_client.login(cfg["base_url"], cfg["username"], cfg["password"])
    except dres_client.DresError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"status": "ok"}

@app.get("/api/dres/current-task")
def dres_current_task():
    """Fetch the current task for the configured DRES_EVALUATION_ID. Requires a prior /api/dres/login."""
    if _dres_session_id is None:
        raise HTTPException(status_code=401, detail="Not logged into DRES - call /api/dres/login first")
    cfg = _dres_config()
    if not cfg["evaluation_id"]:
        raise HTTPException(status_code=400, detail="DRES_EVALUATION_ID not configured")
    try:
        return dres_client.get_current_task(cfg["base_url"], _dres_session_id, cfg["evaluation_id"])
    except dres_client.DresError as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/dres/submit")
def dres_submit(request: DresSubmitRequest):
    """
    Submit an answer for `request.task_id`. `request.payload`'s exact shape
    depends on the task type (KIS/AVS/VQA - see dres_client.submit_answer's
    docstring) and is built by the frontend/caller, not this endpoint -
    kept intentionally generic since the real DRES submission schema isn't
    verified against a live instance yet.

    AVS duplicate-video guard (VBS_GUIDE.md §5.2): if `video_name` is
    provided and already recorded as submitted for this task_id, this is a
    soft warning, not a hard veto - responds 409 WITHOUT calling DRES,
    unless `force=True`, since VBS's live time pressure means the operator
    should be able to override a false alarm instead of being blocked
    outright.
    """
    if _dres_session_id is None:
        raise HTTPException(status_code=401, detail="Not logged into DRES - call /api/dres/login first")
    cfg = _dres_config()
    if not cfg["evaluation_id"]:
        raise HTTPException(status_code=400, detail="DRES_EVALUATION_ID not configured")

    duplicate_warning = _check_avs_duplicate(
        request.task_id, request.video_name, request.force, _avs_submitted_by_task
    )
    if duplicate_warning is not None:
        raise HTTPException(status_code=409, detail=duplicate_warning)

    try:
        result = dres_client.submit_answer(
            cfg["base_url"], _dres_session_id, cfg["evaluation_id"], request.task_id, request.payload
        )
        if request.video_name:
            _avs_submitted_by_task.setdefault(request.task_id, set()).add(request.video_name)
        interaction_log.log_interaction(
            "dres_submit",
            {
                "task_id": request.task_id,
                "payload": request.payload,
                "video_name": request.video_name,
                "result": result,
            },
            dres_config=cfg, session_id=_dres_session_id,
        )
        return result
    except dres_client.DresError as e:
        raise HTTPException(status_code=502, detail=str(e))

# 6. Preprocessing Subprocess Runner

def run_preprocess_sync():
    global _preprocess_process, _preprocess_logs
    try:
        # Start main.py script
        cmd = [
            sys.executable,
            str(WORKSPACE_ROOT / "preprocessing" / "main.py"),
            "--data_dir",
            str(DATASETS_DIR)
        ]
        
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                cwd=str(WORKSPACE_ROOT / "preprocessing")
            )
            
            _preprocess_process = process
            
            # Read stdout line by line and save to log file + memory buffer
            for line in iter(process.stdout.readline, ""):
                _preprocess_logs.append(line.strip())
                # Keep logs buffer to last 1000 lines
                if len(_preprocess_logs) > 1000:
                    _preprocess_logs.pop(0)
                log_f.write(line)
                log_f.flush()
                
            process.wait()
    except Exception as e:
        error_line = f"ERROR executing preprocessing: {str(e)}"
        _preprocess_logs.append(error_line)
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_f:
            log_f.write(error_line + "\n")
    finally:
        _preprocess_process = None

@app.post("/api/preprocess/run")
def start_preprocessing(background_tasks: BackgroundTasks):
    """
    Trigger the preprocessing pipeline in a background task.
    """
    global _preprocess_process, _preprocess_logs
    if _preprocess_process is not None and _preprocess_process.poll() is None:
        return {"status": "already_running", "message": "Preprocessing pipeline is already running."}
        
    _preprocess_logs = ["--- Starting Preprocessing Pipeline ---"]
    if LOG_FILE_PATH.exists():
        try:
            LOG_FILE_PATH.unlink()
        except Exception:
            pass
            
    background_tasks.add_task(run_preprocess_sync)
    return {"status": "started", "message": "Preprocessing pipeline started in background."}

@app.get("/api/preprocess/logs")
def get_preprocess_logs():
    """
    Retrieve live progress logs of the preprocessing pipeline.
    """
    global _preprocess_process
    is_running = _preprocess_process is not None and _preprocess_process.poll() is None
    
    # Read from file as source of truth if logs in memory are empty
    logs = list(_preprocess_logs)
    if not logs and LOG_FILE_PATH.exists():
        try:
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                logs = [line.strip() for line in f.readlines()]
        except Exception:
            pass
            
    return {
        "running": is_running,
        "logs": logs
    }

# 7. Batch Query Subprocess Runner
QUERIES_DIR = WORKSPACE_ROOT / "queries"
BATCH_LOG_FILE_PATH = BACKEND_DIR / "batch_query.log"
_batch_process = None
_batch_logs = []

def run_batch_sync():
    global _batch_process, _batch_logs
    try:
        # Resolve python executable (prioritize venv)
        venv_dir = WORKSPACE_ROOT / "preprocessing" / "venv"
        venv_python = venv_dir / "bin" / "python"
        if not venv_python.exists():
            venv_python = venv_dir / "Scripts" / "python.exe"
        py_executable = str(venv_python) if venv_python.exists() else sys.executable

        cmd = [
            py_executable,
            str(WORKSPACE_ROOT / "inference-code" / "batch_query.py"),
            "--query_file", str(QUERIES_DIR / "queries.json"),
            "--output_dir", str(QUERIES_DIR),
            "--dataset_dir", str(DATASETS_DIR)
        ]
        
        with open(BATCH_LOG_FILE_PATH, "w", encoding="utf-8") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                cwd=str(WORKSPACE_ROOT / "inference-code")
            )
            
            _batch_process = process
            
            for line in iter(process.stdout.readline, ""):
                _batch_logs.append(line.strip())
                if len(_batch_logs) > 1000:
                    _batch_logs.pop(0)
                log_f.write(line)
                log_f.flush()
                
            process.wait()
    except Exception as e:
        error_line = f"ERROR executing batch query: {str(e)}"
        _batch_logs.append(error_line)
        with open(BATCH_LOG_FILE_PATH, "a", encoding="utf-8") as log_f:
            log_f.write(error_line + "\n")
    finally:
        _batch_process = None

@app.get("/api/batch/status")
def get_batch_status():
    """
    Get the status of the batch queries execution, files list in queries directory.
    """
    is_running = _batch_process is not None and _batch_process.poll() is None
    
    files = []
    if QUERIES_DIR.exists():
        for item in QUERIES_DIR.iterdir():
            if item.is_file() and item.suffix.lower() in ['.json', '.csv']:
                files.append({
                    "name": item.name,
                    "size_kb": round(item.stat().st_size / 1024, 2),
                    "modified": item.stat().st_mtime
                })
                
    return {
        "running": is_running,
        "files": files
    }

@app.post("/api/batch/run")
def start_batch_query(background_tasks: BackgroundTasks):
    """
    Trigger the batch queries execution in a background task.
    """
    global _batch_process, _batch_logs
    if _batch_process is not None and _batch_process.poll() is None:
        return {"status": "already_running", "message": "Batch query execution is already running."}
        
    _batch_logs = ["--- Starting Batch Query Processing ---"]
    if BATCH_LOG_FILE_PATH.exists():
        try:
            BATCH_LOG_FILE_PATH.unlink()
        except Exception:
            pass
            
    background_tasks.add_task(run_batch_sync)
    return {"status": "started", "message": "Batch query process started in background."}

@app.get("/api/batch/logs")
def get_batch_logs():
    """
    Retrieve live progress logs of the batch query pipeline.
    """
    global _batch_process
    is_running = _batch_process is not None and _batch_process.poll() is None
    
    logs = list(_batch_logs)
    if not logs and BATCH_LOG_FILE_PATH.exists():
        try:
            with open(BATCH_LOG_FILE_PATH, "r", encoding="utf-8") as f:
                logs = [line.strip() for line in f.readlines()]
        except Exception:
            pass
            
    return {
        "running": is_running,
        "logs": logs
    }

@app.get("/api/batch/results")
def get_batch_results():
    """
    Retrieve the results of the batch query.
    """
    results_path = QUERIES_DIR / "batch_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Batch results file not found.")
        
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Make sure we are in the correct directory when starting
    os.chdir(str(BACKEND_DIR))
    # reload=True only watches cwd by default (webapp/backend/) - explicitly
    # include the shared code directories so edits there trigger a reload too
    reload_dirs = [
        str(BACKEND_DIR),
        str(WORKSPACE_ROOT / "models"),
        str(WORKSPACE_ROOT / "preprocessing"),
        str(WORKSPACE_ROOT / "inference-code"),
    ]
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=reload_dirs, app_dir=str(BACKEND_DIR))

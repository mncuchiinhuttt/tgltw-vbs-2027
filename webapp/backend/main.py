import os
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

# 1. Path Configuration
BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent.parent
DATASETS_DIR = WORKSPACE_ROOT / "datasets"
LOG_FILE_PATH = BACKEND_DIR / "preprocessing.log"

# Add directories to sys.path to load config and models
sys.path.append(str(WORKSPACE_ROOT))
sys.path.append(str(WORKSPACE_ROOT / "inference-code"))

# Safe to import at module level (unlike config/models/search.hybrid_search,
# which the endpoints import lazily): search/ is a namespace package and
# session_history.py has no imports of its own at all, so this pulls in no
# torch/qdrant/config side effects.
from search.session_history import trim_history

app = FastAPI(title="Multimedia Retrieval API", version="1.0.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    from models.embedding import QwenVL8BEmbedder, DashScopeCloudEmbedder
    if config.EMBEDDING_OPTION == "local":
        return QwenVL8BEmbedder()
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

        if _vlm is None:
            print("Initializing VLM...")
            _vlm = load_vlm()

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
        visual_count = 0
        audio_count = 0
        
        # Check visual_index
        try:
            visual_info = client.get_collection(collection_name="visual_index")
            visual_count = visual_info.points_count
        except Exception:
            pass
            
        # Check audio_env_index
        try:
            audio_info = client.get_collection(collection_name="audio_env_index")
            audio_count = audio_info.points_count
        except Exception:
            pass
            
        return {
            "status": "connected",
            "host": config.QDRANT_HOST,
            "port": config.QDRANT_PORT,
            "visual_points": visual_count,
            "audio_points": audio_count
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

def _reset_session_state():
    """
    Clears every per-task field of the single global interactive session.
    Pure dict mutation, factored out of the endpoint so the reset semantics
    stay in one place - `_session_state` is rebound field-by-field rather
    than replaced, so any module holding a reference to the same dict (or to
    _session_state["history"], which record_feedback_in_history/trim_history
    mutate in place) observes the reset.
    """
    _session_state["history"].clear()
    _session_state["last_query_vector"] = None
    _session_state["pending_clarification"] = None
    _session_state["last_candidate_info"] = {}


@app.post("/api/session/reset")
def reset_session():
    """
    Drops the conversational session (CQR history, Rocchio base vector,
    pending clarification, candidate cache) so the NEXT search starts clean.

    Required because one backend process serves a whole VBS session of many
    consecutive tasks: without this, turn 1 of a new task is CQR-rewritten
    against the previous task's queries, which corrupts exactly the mode
    (KIS-C) that depends on history most. Deliberately operator-triggered
    rather than driven off DRES task changes - dres_client.get_current_task's
    response fields are still unverified against a live instance, so keying
    an automatic reset on `task_id` would make correctness depend on a field
    that may not exist.
    """
    turns = len(_session_state["history"])
    _reset_session_state()
    return {"status": "ok", "cleared_turns": turns}


@app.post("/api/search")
async def run_search(request: SearchRequest):
    """
    Execute Type 1 (Textual-KIS), Type 2 (VQA), or Type 3 (Temporal-Alignment) search.
    """
    if request.type not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="Invalid search type. Must be 1, 2, or 3.")
        
    try:
        # Initialize services dynamically
        query_proc, searcher, reranker = init_services(query_type=request.type)
        
        # Determine dataset_dir
        dataset_dir = request.dataset_dir or str(DATASETS_DIR)

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
        # Cap the history the NEXT rewrite_query_cqr call has to read. Every
        # turn is rendered verbatim into the few-shot prompt, so an untrimmed
        # session grows both the prompt and the rewrite latency without bound
        # on a 7-minute clock. Keeps turn 1 (the task's opening description).
        trim_history(_session_state["history"])

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
            from search.kis_c_scoring import boost_by_clarification_answer

            # KIS-C clarification-answer boost (Sekulic et al. arXiv:2008.03717):
            # additive re-rank of the CURRENT turn's normal RRF-fused pool
            # using whichever of last turn's clarification candidates now
            # overlap with the operator's answer - not a fast path that
            # skips search, the full pipeline above still ran. Run before
            # the ambiguity check: a discriminating answer sharpens the
            # top-1/top-2 margin, which lowers the new ambiguity score and
            # stops the system asking a redundant second question.
            if pending_clarification and (request.clarification_answer or "").strip():
                candidates = boost_by_clarification_answer(
                    candidates,
                    pending_clarification.get("candidate_ids") or [],
                    request.clarification_answer,
                )
                clarification_boost_applied = True

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
                _session_state["pending_clarification"] = {
                    "question": clarification_question,
                    "candidate_ids": summary_ids,
                }

            top_candidates = reranker.rerank_type1(resolved_query, candidates[:10], verify=request.verify)
            top_candidates = [
                c for c in top_candidates
                if c.get("rerank_score", 0.0) >= config.RERANK_SCORE_THRESHOLD
            ]
            for idx, c in enumerate(top_candidates):
                results.append({
                    "rank": idx + 1,
                    "score": c.get("rerank_score", 0.0),
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
            top_candidates = reranker.rerank_type2_vqa(
                resolved_query, sub_queries, candidates[:10], dataset_dir, verify=request.verify
            )
            
            for idx, c in enumerate(top_candidates):
                # Retrieve concise answer for the top candidate
                answer = "N/A"
                if idx == 0:
                    # In backend we try to extract the real cropped frame for VLM answer if possible
                    try:
                        from PIL import Image
                        import os
                        video_name = c["payload"]["source_file"]
                        timestamp = c["payload"]["timestamp"]
                        frame_path = os.path.join(dataset_dir, video_name)
                        frame_img = None
                        if os.path.exists(frame_path) and frame_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                            frame_img = Image.open(frame_path).convert("RGB")
                        else:
                            # Try video extraction
                            cap = cv2.VideoCapture(frame_path)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            if fps > 0:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
                                ret, frame = cap.read()
                                if ret:
                                    frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                cap.release()
                        
                        answer_prompt = f"Answer the following question about this image: {resolved_query}. Be concise."
                        vlm = load_vlm()
                        answer = vlm.generate(frame_img, answer_prompt).strip()
                    except Exception as e:
                        print(f"Failed to generate answer for top result: {e}")
                        answer = "Error generating answer"
                
                results.append({
                    "rank": idx + 1,
                    "score": c.get("final_score", 0.0),
                    "vqa_score": c.get("vqa_score", 0.0),
                    "rrf_score": c.get("rrf_score", 0.0),
                    "id": c["id"],
                    "payload": c["payload"],
                    "answer": answer if idx == 0 else None,
                    "matched_via": c.get("matched_via", [])
                })
            # Record the generated answer against this turn so a later CQR
            # rewrite (e.g. "was there a sign in that scene too?") can
            # resolve against what the system actually answered, not just
            # the query text.
            if results and results[0].get("answer"):
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
    try:
        query_proc, searcher, reranker = init_services(query_type=2)
        dataset_dir = request.dataset_dir or str(DATASETS_DIR)

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

        results = [
            {
                "rank": idx + 1,
                "score": c.get("final_score", 0.0),
                "vqa_score": c.get("vqa_score", 0.0),
                "rrf_score": c.get("rrf_score", 0.0),
                "id": c["id"],
                "payload": c["payload"],
            }
            for idx, c in enumerate(top_candidates)
        ]
        return {"query": request.query, "video_name": request.video_name, "results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"In-video search failed: {str(e)}")

@app.get("/api/media/frame")
def get_frame(video_name: str, timestamp: float):
    """
    Dynamically extract a frame from a video at a specific timestamp and return as JPEG.
    """
    video_path = DATASETS_DIR / video_name
    if not video_path.exists():
        # Fallback to image check
        if video_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            return FileResponse(str(video_path))
        raise HTTPException(status_code=404, detail="Media file not found")

    # If it is a static image, return directly
    if video_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        return FileResponse(str(video_path))

    # Read video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Unable to retrieve FPS from video")

    frame_idx = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        # Create an empty dark gray PIL Image
        img = Image.new("RGB", (640, 360), color=(30, 41, 59))
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return Response(content=output.getvalue(), media_type="image/jpeg")

    # Encode to JPEG
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/api/media/video/{video_name}")
def get_video(video_name: str):
    """
    Serve video file directly with support for seeking/range queries.
    """
    video_path = DATASETS_DIR / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
        
    return FileResponse(str(video_path))

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
        
        with open(LOG_FILE_PATH, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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
        with open(LOG_FILE_PATH, "a") as log_f:
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
            with open(LOG_FILE_PATH, "r") as f:
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
        
        with open(BATCH_LOG_FILE_PATH, "w") as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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
        with open(BATCH_LOG_FILE_PATH, "a") as log_f:
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
            with open(BATCH_LOG_FILE_PATH, "r") as f:
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=reload_dirs)

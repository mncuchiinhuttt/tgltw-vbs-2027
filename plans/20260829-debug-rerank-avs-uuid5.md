# Debugging & Root Cause Analysis Plan: Webapp Rerank Scope, AVS Lane & Preprocessing Idempotency

> **Target Repository**: `tgltw-vbs-2027`
> **Date**: 2026-08-29

---

## 1. Issue 1: Webapp Backend Rerank Slice Scope (`RERANK_TOP_K`)

### Root Cause
In `webapp/backend/main.py`:
- Lines 578 & 604 slice `candidates[:10]` directly instead of honoring `config.RERANK_TOP_K` (default 20).
- Unlike `batch_query.py` and `run_eval.py`, the webapp dropped candidates beyond the slice rather than appending the remaining candidates in their original retrieval order via `rerank_with_tail()`.

### Solution
- Use `config.RERANK_TOP_K` and `reranker.rerank_with_tail()` for both Type 1 and Type 2 searches.
- Preserve the full candidate pool while bounding VLM reranking cost to the top-K head.

---

## 2. Issue 2: Dedicated AVS (Type 4) Search Lane

### Root Cause
- In `webapp/backend/main.py`, requests with `type == 4` (AVS / Ad-hoc Video Search) had no dedicated branch in `/api/search` and fell through or executed single-item KIS reranking, which hurts diverse cross-video shot discovery.

### Solution
- Add an explicit `elif request.type == 4:` block in `webapp/backend/main.py`.
- Apply `searcher.diversify_by_scene(top_k=SUBMISSION_TOP_K)` to prioritize distinct shot and video diversity without incurring unneeded single-frame VLM rerank latency.
- Format results consistently for frontend display and DRES batch submission.

---

## 3. Issue 3: Preprocessing Idempotency with Stable UUID5

### Root Cause
In `preprocessing/main.py`:
- Speech transcript points (`indexer.index_visual_point(str(uuid.uuid4()), speech_vector, payload)`) and CLAP ambient audio points (`indexer.index_audio_point(str(uuid.uuid4()), clap_vector, audio_payload)`) generate random UUID4s.
- Re-running or resuming the preprocessing pipeline creates duplicate points in Qdrant collections.

### Solution
- Replace random `uuid.uuid4()` with deterministic `uuid.uuid5(uuid.NAMESPACE_URL, ...)`:
  - Speech ASR: `uuid5(NAMESPACE_URL, f"vbs-speech:{video_name}:{start_t:.2f}_{end_t:.2f}")`
  - Audio CLAP: `uuid5(NAMESPACE_URL, f"vbs-audio:{video_name}:{start_sec:.2f}_{end_sec:.2f}")`
  - Standalone frames: `uuid5(NAMESPACE_URL, f"vbs-frame:{video_name}:{frame_idx or timestamp}")`

---

## 4. Verification Plan
- Unit tests validating `RERANK_TOP_K` behavior in webapp backend.
- Unit tests verifying Type 4 AVS diverse ranking output.
- Unit tests verifying deterministic, idempotent point ID generation.
- Full pytest test suite run.

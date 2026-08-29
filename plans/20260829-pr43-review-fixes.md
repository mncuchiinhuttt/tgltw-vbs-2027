# Technical Implementation Plan: PR #43 Quality & Defect Remediation

> **Target Repository**: `tgltw-vbs-2027`
> **PR**: `#43`
> **Date**: 2026-08-29

---

## 1. Overview & Root Cause Analysis

Following the 16-agent comprehensive code review on PR #43, 3 functional subsystems contain critical bugs and contract violations that need surgical remediation:

1. **`queries/run_vbs_audit.py`**:
   - `rerank_type3_temporal` signature mismatch (`(events, candidates, top_k=top_k)` vs `(query_text, candidates, query_proc, searcher)`).
   - Result sequence unpacking reads missing `payload` instead of `video_name` / `frame_ids`.
   - `extract_vqa_answer` calls non-existent `vlm.answer_question` and fails to load visual keyframe images.
   - `submission.zip` bundles internal `.details/*.json` trace files instead of strictly clean `.csv` files.
   - JSON query manifest loading ignores `parse_query_type` when `"type"` key is absent.

2. **`evaluation/run_eval.py`**:
   - `use_priors` parameter / `--with_priors` CLI flag is received but completely unreferenced in retrieval/scoring.
   - Type 3 timestamp distance crashes with `TypeError` when ground-truth timestamp is a string; missing `frame_id` coordinate support.
   - VQA Exact Match bidirectional substring search (`gen_ans in gt_ans`) causes false positives when fallback `"N/A"` matches words like `"banana"`.

3. **`tests/test_vbs_audit.py`**:
   - `test_apply_audit_priors_disabled_via_env` sets `os.environ["VBS_DISABLE_AUDIT_PRIORS"] = "1"` without restoring environment in `tearDown` or `addCleanup`.
   - Multi-event parsing regex in `parse_trake_events` fails on single-line inputs.

---

## 2. Phased Implementation Strategy

### Phase 1: `queries/run_vbs_audit.py`
- Fix `rerank_type3_temporal` call to `(q_text, candidates[:top_k], query_proc, searcher)`.
- Extract sequence predictions using `seq.get("video_name")` and `seq.get("frame_ids", [])`.
- Implement frame-grounded VQA answering using `load_submission_frame` and `vlm.generate(frame_img, prompt)` with fail-closed fallback to `"N/A"`.
- Update `submission.zip` packaging to include ONLY `*.csv` files and exclude all `.details` files.
- Ensure `parse_query_type` is invoked during query loading for all items lacking explicit `type`.

### Phase 2: `evaluation/run_eval.py`
- Apply `apply_audit_priors` inside `run_benchmark` when `use_priors` is True.
- Ensure safe float conversions for timestamps and support `frame_id` matching for Type 3.
- Fix VQA Exact Match to require normalized string equality (`gen_ans == gt_ans`) and reject `"N/A"` substring matching.

### Phase 3: `tests/test_vbs_audit.py`
- Add `setUp` / `tearDown` or `self.addCleanup` to isolate environment variables.
- Extend `parse_trake_events` regex to handle inline multi-event markers (`E1: ... E2: ...`).
- Add tests validating CSV structure and clean `.zip` contents without `.details`.

### Phase 4: Full Validation & Test Execution
- Run `pytest tests/` in `tgltw-vbs-2027/.venv`.
- Run mock audit run with `queries/run_vbs_audit.py` and benchmark evaluation.

### Phase 5: Git Commit & Push
- Commit all fixes with clear conventional commit messages and push to `origin/main`.

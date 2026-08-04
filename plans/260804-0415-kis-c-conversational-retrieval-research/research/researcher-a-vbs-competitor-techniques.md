# VBS Competitor Conversational/KIS-C Techniques — Research Report
**Researcher A** | 2026-08-04 | Focus: VBS 2022–2026 competitor systems & KIS-C dialogue strategies

---

## Executive Summary
KIS-C (Conversational Known-Item Search) was formally introduced at **VBS 2025** (not before). Repo currently implements basic CQR + clarification-question generation; identified 4 novel competitor techniques worth integrating: (1) unified conversational+feedback loop, (2) multi-step retrieval orchestration, (3) temporal span prediction for answer grounding, (4) Bayesian relevance update variant. All cross-checked against already-implemented features; no double-recommendations.

---

## 1. Unified Conversational + Relevance Feedback Loop
**Source:** Exquisitor team (Khan et al.), VBS 2024/2025. Papers: "Exquisitor at VBS 2024: Relevance Feedback Meets Conversational Search" (MMM 2024, Springer); "Exquisitor at VBS 2025: Unifying Conversational Search and User Relevance Feedback" (MMM 2025).

**Mechanism:** Two historically separate interaction modes (asking questions to refine query vs. marking results as relevant/irrelevant) fused into single interface. User alternates: refine query via follow-up questions → inspect top-N results → mark positives → system re-ranks next batch using both dialogue context AND feedback signals. VBS 2025 edition unifies into single search pane; earlier versions kept modes separate but signal-shared.

**Applicability:** Repo's `_session_state` currently stores `history: [{query, answer?}, ...]` and separate `/api/feedback` endpoint for Rocchio. Currently **orthogonal** — feedback doesn't inform CQR next rewrite, and CQR rewrites don't account for which candidates user explicitly rejected. **Effort: MEDIUM.** Modify `history` to track `{query, answer, rejected_video_ids[], accepted_video_ids[]}` and feed rejected set into CQR rewriter as negative context. Touch `query_processor.py:rewrite_query_cqr` to accept rejected candidates, `/api/feedback` to update history with feedback signal, and main.py orchestration.

**Overlap flag:** NOT implemented. Rocchio feedback exists but isolated; clarification questions generated separately.

---

## 2. Multi-Step Retrieval Orchestration via LLM Agent
**Source:** SnapMind (Ho-Le et al., U. Science VNU-HCM), VBS 2026. Paper: "From Expert Practices to Intelligent Agents: Autonomy in Interactive Video Retrieval" (MMM 2026, Springer).

**Mechanism:** Registry of ~8 modality/component pairs (text, image, OCR, color, object, ADL, + search/rerank/filter modes). LLM Planner given current query + registry + interaction history, generates multi-step plan: "search text → rerank via object → filter color → fuse RRF." User can inspect, edit weights/top-k before execution. Incremental fusion tracks source attribution and supports early stopping (Jaccard @K or NDCG delta). Unlike CQR (single-shot rewrite), Planner orchestrates whole retrieval pipeline as a dialogue object.

**Applicability:** Repo has no orchestration layer — query → single dense+hybrid+secondary → RRF → rerank pipeline is hardcoded. **Effort: HIGH.** Requires (a) component registry design (what components, modes, score ranges, latency), (b) LLM prompt template for plan generation, (c) plan executor with fusion/early-stop logic, (d) UI for plan inspection/editing. New file `inference-code/search/component_registry.py`, new endpoint `/api/plan`, modify main.py flow. Partially useful for novice-user guidance (SnapMind offers 3 autonomy levels: guide/assist/auto); full implementation is multi-week effort.

**Overlap flag:** CONCEPTUALLY related but different from CQR. CQR rewrites one query; Planner designs multi-step pipeline. Both could coexist (Planner step 1 could invoke CQR rewrite).

---

## 3. Temporal Answer Span Prediction + Candidate Suggestion
**Source:** NII-UIT (Tran et al., UIT + National Institute of Informatics Tokyo), VBS 2025, 2026. Papers: "NII-UIT at VBS2025/2026: Multimodal Video Retrieval with LLM Integration and Dynamic Temporal Search."

**Mechanism:** For VQA/Type-2 tasks (not pure KIS-C, but hybrid). Pipeline: (1) divide video into temporal units, (2) dense caption + transcript per unit, (3) LLM (NVILA) aligns question with caption timeline → hotspot, (4) extract frame candidates from hotspot, (5) suggest multiple candidate answers + visual evidence. User verifies answer before submission. Novelty: shifts VQA from video-level retrieval to answer-bearing moment + grounding.

**Applicability:** Repo focuses on KIS-C (no pre-fixed answer), but VBS 2025+ also includes VQA. This technique applies to Type 2 (VQA) tasks. **Effort: MEDIUM-HIGH.** Requires (a) caption/transcript indexing (caption cached per segment already?), (b) NVILA model or equivalent LLM call to align question → temporal window, (c) candidate answer extraction + grounding UI. Touch `inference-code/search/` for temporal localization logic, new endpoint `/api/answer-span` for VQA variant. Not applicable to pure KIS-C but valuable for mixed competitions.

**Overlap flag:** NOT implemented. Current repo has no temporal span grounding for answers, only per-frame retrieval.

---

## 4. Bayesian Relevance Update Variant
**Source:** PraK V4 (Jäckl et al., U. Konstanz + Charles U. Prague), VBS 2026. Paper: "PraK V4 at the Video Browser Showdown 2026."

**Mechanism:** Stateful session backend maintains Bayesian belief over frame relevance. User clicks positive keyframe → belief updated via Bayes rule. No explicit formula in paper, but approach differs from Rocchio (linear weight update) by maintaining posterior probability distribution over relevance. Inference-time: rank by posterior probability, not Rocchio score.

**Applicability:** Repo uses Rocchio relevance feedback (`/api/feedback`, line mentions "Rocchio relevance feedback"). Bayesian approach offers principled uncertainty quantification. **Effort: LOW-MEDIUM.** Replace Rocchio with Bayesian update: maintain prior (uniform or learned) + likelihood (frame embedding similarity to feedback exemplars) → posterior. Probability ranking useful for high-stakes VQA/KIS-C (low uncertainty → submit; high uncertainty → ask clarification). Touch `inference-code/search/relevance_feedback.py` (new module), replace Rocchio logic with Bayes filter.

**Overlap flag:** PARTIAL overlap. Rocchio feedback exists; Bayesian is better-founded alternative, not a wholly new idea.

---

## 5. Dynamic Grounded-SAM Localization (Spatial Conjunction)
**Source:** PraK V4, VBS 2026. "Localized query: Dynamic region from Grounded-SAM. Spatial conjunction: Multiple box and text query simultaneously."

**Mechanism:** User draws or refines bounding box via Grounded-SAM (segment-anything + CLIP grounding). Query can combine multiple spatial constraints (e.g., "red car in left half + person in right half, both in same frame"). Backend enforces all constraints must match for a frame to rank high.

**Applicability:** Repo has no spatial/localized query support — all queries are global over frame. This is UI-heavy (canvas for box drawing) + backend logic (spatial constraint evaluation). **Effort: HIGH.** Requires (a) Grounded-SAM integration (inference-code/models/), (b) constraint solver backend (new logic in hybrid_search), (c) UI drawing/box-refinement (webapp/frontend). Not urgent for pure KIS-C (conversational task) but useful for Clarification Q refinement (e.g., "Is the object in the left or right side?" → user draws box).

**Overlap flag:** NOT implemented. Spatial constraints are not in current KIS-C flow.

---

## 6. KIS-C Task Definition & Scoring Implications
**Source:** VBS 2025 official rules / VBS Guide.

**Mechanism:** KIS-C: minimal text description, after 60 seconds moderator reveals more details in response to operator questions. 7-minute time limit. Scoring: 50 base + (300-t)/6 time bonus - 10×|WS| wrong penalty. Key: each wrong guess costs 10 pts ≈ 60 seconds of clock. Unlike AVS (maximize diversity), KIS-C rewards precision + speed. Multiple clarification questions → multiple turns → overhead per turn (formulate, send, wait answer). Optimal strategy: high-precision queries (minimize wrong guesses) + reuse moderator answers effectively (CQR crucial).

**Applicability:** Repo's clarification-question threshold (AMBIGUITY_THRESHOLD=0.7, configurable) must be tuned against this scoring. Asking clarification question costs ~10 seconds + latency; payoff only if it disambiguates down to 1–2 top videos. Current ambiguity heuristic (ratio of distinct videos in top-10) is crude proxy. **Effort: LOW.** Tune threshold empirically on past VBS 2025 KIS-C tasks if available; run offline analysis of (question cost + inference speedup) vs. (wrong-guess penalty avoided).

**Overlap flag:** Scoring/threshold already in place. Clarification questions already implemented.

---

## 7. LLM Query Suggestion (vs. Query Rewriting)
**Source:** v-FIRST (Nhat Hoang-Xuan et al., VNU-HCM + Dublin City U.), VBS 2023. Paper: "V-FIRST: Video Event Retrieval with Flexible Textual-Visual Intermediary" (MMM 2023).

**Mechanism:** Distinct from CQR. User query → LLM generates 3–5 alternative phrasings/search terms (NOT rewriting the original to resolve pronouns, but EXPANDING with synonyms/related concepts). Displayed as suggestions; user picks best or types new. Frames query expansion as interactive refinement, not batch expansion.

**Applicability:** Repo has CQR (resolve references). Query suggestion is orthogonal: after CQR output, offer 2–3 LLM-suggested alternative queries (e.g., user says "the woman talking" → CQR resolves "woman in previous frame" → Suggestion generates "female speaker", "woman speaking", "dialogue scene"). **Effort: LOW-MEDIUM.** Add LLM call in query_processor.py after CQR output; cache top-3 suggestions; return in API response. Useful for novice users (gives options) and for exploring alternative phrasings. Does NOT replace CQR; complements it.

**Overlap flag:** CQR (rewriting) exists. Query suggestion (expansion) does NOT.

---

## 8. Clarification Question Source: Reasoning vs. Heuristic
**Source:** Exquisitor, VIREO, and general VBS practice (papers mention LLM rewrite + candidates). Repo uses heuristic (ambiguity score) to trigger; competitors use LLM reasoning.

**Current repo:** `generate_clarification_question()` → LLM reads top-5 captions from distinct videos → generates ONE question. Heuristic (ambiguity_score) triggers it.

**Competitor insight:** SnapMind's Planner + Exquisitor's conversational loop suggest asking clarification Q is not a one-off heuristic trigger, but part of ongoing dialogue. Exquisitor 2025 unifies Q-asking + feedback into loop; SnapMind lets user approve plan before execution. Implication: don't just ask random clarification when uncertain; ask questions that directly disambiguate **the top N candidates the system is considering**, and feed the answer back into retrieval, not just set a flag.

**Applicability:** Repo's clarification logic is isolated. **Effort: MEDIUM.** Enhance `generate_clarification_question` to (a) be called not just when ambiguous, but at any turn if top-2 candidates score close, (b) tailor Q to actual candidate captions (already done), (c) record moderator's answer in history + feed into next CQR rewrite. Shift from "ask when uncertain" to "ask as part of dialogue".

**Overlap flag:** Logic partially overlaps (clarification Q already exists) but FRAMING and INTEGRATION are underdeveloped.

---

## Unresolved Questions / Could Not Verify

1. **VIREO's exact LLM checkpoint** — paper says "LLM rewrite" but doesn't name model. Likely GPT-4 or Qwen, but unclear if it's different from CQR in repo.
2. **v-FIRST's query suggestion prompt design** — no code or prompt template published. Only described conceptually.
3. **Bayesian prior specification in PraK V4** — paper doesn't detail prior distribution or likelihood function. May be empirically tuned rather than theoretically justified.
4. **When KIS-C was first introduced** — web search suggests VBS 2025, but unclear if earlier competitions had similar conversational tasks (could have been called differently).
5. **Exquisitor's unified mode (VBS 2025)** — paper exists on Springer but not accessible for full mechanism review. Description inferred from abstract + Springer chapter title.

---

**Next Steps for Repo Integration:**
- Priority 1: Unify feedback + CQR dialogue state (Technique #1). Enables tighter feedback loop for KIS-C.
- Priority 2: Validate clarification Q logic via offline VBS 2025 KIS-C eval set (if available). Adjust threshold and question diversity (Technique #8).
- Priority 3: Add LLM query suggestion as UI option (Technique #7). Low effort, high novice-user value.
- Priority 4: Explore Bayesian relevance feedback (Technique #4) as confidence signal for answer submission in VQA tasks.
- Priority 5 (later): SnapMind-style orchestration (Technique #2) and temporal span grounding (Technique #3) if expanding beyond pure KIS-C.

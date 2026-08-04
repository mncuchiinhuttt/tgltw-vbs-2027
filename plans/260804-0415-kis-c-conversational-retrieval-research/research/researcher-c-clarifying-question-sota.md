# SOTA Clarifying Question Generation & Preference Elicitation for Interactive Video Search

## 1. IR Benchmarks & Datasets for Clarification

### Qulac (arXiv:1907.06554, SIGIR 2019)
**Source:** Aliannejadi, Zamani, Crestani, Croft. "Asking Clarifying Questions in Open-Domain Information-Seeking Conversations."  
**Mechanism:** 10K Q-A pairs over 198 TREC topics + 762 facets. Offline eval methodology. NO ranking/scoring mechanism for candidate questions—just labeled clarifications. Oracle shows asking *one good question* → 170% P@1 improvement, but paper doesn't specify how to SELECT the good one.  
**Time-pressure fit:** MEDIUM. Methodologically sound but designed for batch research, not real-time. Dataset valuable for training but current repo is training-free.  
**Applicability:** Benchmark reference only. No direct code integration. Guides what "good questions" should target (TREC facets: named entities, properties, relationships).

### ClariQ (ConvAI3 2020, ACL Anthology)
**Source:** Aliannejadi et al. "ConvAI3: Generating Clarifying Questions for Open-Domain Dialogue Systems."  
**Mechanism:** ~1K+ conversations, multi-turn. Competition-style task. Stage 1 (offline) ranks given clarifications; Stage 2 (online) generates them. Proposes "pipeline consisting of offline and online steps" but specifics thin in available abstracts.  
**Time-pressure fit:** MEDIUM. Multi-turn design adds latency. Useful for evaluation protocol (how to score question quality) rather than generation strategy.  
**Applicability:** Evaluation framework reference. Could adapt Stage 1 ranking methodology to score N candidate questions before showing one, but requires training data.

### ClarQ (arXiv:2006.05986, ACL 2020)
**Source:** Kumar & Black. "ClarQ: A large-scale and diverse dataset for Clarification Question Generation."  
**Mechanism:** 2M examples across 173 StackExchange domains. Neural 2-step classifier (precision→recall bootstrapping). No explicit info-gain/ranking of question alternatives; dataset is source data for finetuning generation models.  
**Time-pressure fit:** LOW. Requires model finetuning on 2M examples; repo cannot afford training pipeline.  
**Applicability:** Training data source only if deciding to finetune. Does not solve the "which question to generate" problem for prompt-based approach.

### MIMICS (arXiv:2006.10174, SIGIR 2020)
**Source:** Zhang et al. "MIMICS: A Large-Scale Data Collection for Search Clarification."  
**Mechanism:** 600K+ Bing search log clarifications (3 subsets: Click, ClickExplore, Manual). Each query→clarifying question + 5 candidate answers. Real-world scale + quality annotations (MIMICS-Manual). No explicit ranking among questions; dataset is for training/evaluating systems that DECIDE WHEN to clarify, not which question to pick.  
**Time-pressure fit:** MEDIUM for benchmarking, LOW for real-time generation. Large, real-world but pre-collected questions, not generation strategy.  
**Applicability:** Could extract candidate captions structure (question + 5 answers) as model for current repo's 5-candidate approach. Evaluation set for comparing generated questions against human baselines.

---

## 2. Facet-Driven & Contrastive Question Generation

### Facet-Driven Clarification (Sekulic et al., SIGIR-TIIR 2021)
**Source:** Sekulic, Aliannejadi, Crestani. "Towards Facet-Driven Generation of Clarifying Questions for Conversational Search."  
**Mechanism:** Extract facets (distinct query aspects) from top-retrieved docs. Fine-tune GPT-2 conditioned on predicted facets to generate targeted questions. Zero-shot variant (Wang et al., WWW 2023, arXiv:2301.12660) uses facet-constrained prefix prompting—no finetuning needed. Prefix+facets guide LLM to ask about specific aspect of ambiguous candidates.  
**Time-pressure fit:** HIGH (zero-shot variant). Single LLM call + facet extraction from existing candidate captions = ~200ms. Facets derivable from caption text (entities, attributes already in payload). No training required.  
**Applicability:** Direct extension of `generate_clarification_question`. Replace current generic prompt with facet-extraction step: (1) analyze top-5 captions for differentiating facets (color, action, location, etc.); (2) prefix LLM prompt with facet + instruction "generate a question about [FACET]" that disambiguates. E.g., if 5 videos differ in vehicle color, ask about color, not generic "what happens in the clip?"

### Contrastive Visual Question Generation (ConVQG, arXiv:2402.12846, 2024)
**Source:** Li et al. "ConVQG: Contrastive Visual Question Generation with Multimodal Guidance."  
**Mechanism:** Generate questions via DIFF between candidates. Contrastive learning: questions grounded on positive candidate set (multimodal)—discriminate from negatives. Dual-objective loss for both text and image alignment.  
**Time-pressure fit:** MEDIUM-HIGH. Contrastive training typically baked into model, but concept applicable: generate N candidate questions, score each by how much it SPLITS current candidate set (information gain proxy).  
**Applicability:** Concept: "Learning to Disambiguate by Asking Discriminative Questions" (Li et al., ICCV 2017, arXiv:1708.02760) provides discriminative question generation—train or prompt LLM to generate question that isolates one candidate from others by contrasting attributes. E.g. "Candidate A has a red car, B/C/D/E don't—ask about car color." Implementable as: (1) compute attribute DIFF between ambiguous candidates; (2) prompt: "Generate ONE short question that would tell apart videos where [DIFF_ATTRIBUTE] varies"; (3) score output by "does this question mention the differentiating attribute?"

### Referring Expression Generation (Visual Grounding)
**Source:** Multiple authors on RE generation via attributes (Mao et al., Daniluk et al., etc.). Covers visual grounding for disambiguation.  
**Mechanism:** Referring expressions disambiguate objects via discriminative attributes. REG task: given object + scene, generate unambiguous description that distinguishes target. Same principle applicable to video captions: generate description (question) that refers uniquely to one candidate video.  
**Time-pressure fit:** MEDIUM. REG typically trained; prompt-based variant: provide candidate captions + ask LLM "what single visual attribute or action uniquely identifies video [X] vs the others?" Then convert to question.  
**Applicability:** Enhance caption analysis in `generate_clarification_question`: parse candidate captions for atomic attributes (objects, actions, colors, locations), compute which attributes are *shared* vs *unique*, generate question about a unique attribute.

---

## 3. Information Gain & Question Selection

### Maximum Information Gain Selection (arxiv:1911.03598, arXiv:2507.06467)
**Source:** "Interactive Classification by Asking Informative Questions"; "Interactive Text-to-SQL via Expected Information Gain for Disambiguation."  
**Mechanism:** Rank candidate questions by expected information gain: `IG(q) = H(Y) - E[H(Y|answer_to_q)]`, where H = entropy over candidate set, E[...] averages over possible answers. SELECT question maximizing entropy reduction. "Look-before-ask" principle: sample candidate questions, score by information gain, pick top-1.  
**Time-pressure fit:** MEDIUM-HIGH. Entropy calc is cheap: O(N²) for N candidates. Generate M candidate questions (e.g., M=3-5), score each by how much splitting the 5-candidate set, pick highest. Total: single LLM call for generation + lightweight scoring.  
**Applicability:** NEW function alongside `generate_clarification_question`. (1) `generate_n_candidate_questions(captions, n=5)`: prompt LLM to generate 5 different clarifying questions; (2) `score_question_by_information_gain(question, captions)`: for each candidate question, estimate P(each candidate matches the question's topic) via LLM or caption keyword overlap; compute entropy reduction; (3) pick question with max IG. Integrate into `hybrid_search.py:compute_ambiguity_score()` flow—compute IG score *before* firing clarification.

### No empirical IR study directly compares IG-ranked vs single-shot clarification under VBS time pressure (unresolved).

---

## 4. Structured Answer Incorporation & Re-ranking

### Clarification Answer → Direct Re-rank (Sekulic et al., arXiv:2008.03717)
**Source:** "Analysing the Effect of Clarifying Questions on Document Ranking in Conversational Search."  
**Mechanism:** When user answers clarification (e.g., "yes, red car"), treat as constraint/filter on existing candidate set. Re-rank in-place: boost videos matching answer, reorder top-10. Showed 18% recall, 12% nDCG@3 improvement vs baseline. Key: *don't re-run dense search*—use existing candidate list.  
**Time-pressure fit:** HIGH. Constraint application + re-rank = ~50ms. No model inference.  
**Applicability:** Add `clarification_answer_rerank()` function in `hybrid_search.py`. Input: (candidate_list_from_ambiguity_check, user_answer_string, original_question). Parse answer for keywords/entities; boost candidates matching keywords; re-sort. E.g., if Q="what color is the car?" and answer="red", boost all candidates with "red" in caption/OCR/object tags.

### Iterative Clarification & Rewriting (ICR, arXiv:2509.05100)
**Source:** "ICR: Iterative Clarification and Rewriting for Conversational Search."  
**Mechanism:** Alternate: generate clarification question → use answer to REWRITE query → new search. Better than single-shot. SOTA on CAsT, TREC CaRE benchmarks.  
**Time-pressure fit:** LOW for live VBS. Multiple LLM calls (clarify + rewrite + new dense search) = 1-2 seconds latency. Acceptable for offline analysis, not for competition timed turns.  
**Applicability:** Optional follow-up after initial answer. If user's clarification answer is rich (not just yes/no), feed to `rewrite_query_cqr()` in `query_processor.py` for next iteration. But baseline should be fast in-place re-rank (above).

### Multiple-Choice Clarification vs Open-Text
**Source:** MCMIPL (Zhang et al., 2022, arXiv:2112.11775); Survey on Conversational Recommenders (arXiv:2004.00646).  
**Mechanism:** Multiple-choice (MC) questions elicit preference more efficiently than open or binary-yes/no. MC can encode candidate videos as choices: "Is it clip A, B, C, D, or E?". Critiquing-based recommenders show attribute-based elicitation (not item-based) works early, item-based later.  
**Time-pressure fit:** MEDIUM. MC requires structuring answers (link to candidate video IDs). VBS's actual setup: operator relays question *verbally* to moderator who picks from candidate set mentally—system doesn't parse the answer programmatically anyway, moderator does. So MC doesn't reduce *operator's* interpretation time. Open text question may be more natural for verbal relay. *Not recommended* for VBS unless backend refactored to parse structured answers.  
**Applicability:** Current repo architecture treats answer as free-text query input, not structured choice. Skip MC for now; focus on open-text discriminative questions with in-place re-ranking.

---

## 5. Unresolved / Couldn't Verify

- **Empirical latency comparison**: No head-to-head <150ms benchmark comparing (a) current single-shot generation, (b) multi-candidate + IG selection. Recommend pilot test: measure time to generate 5 candidate questions + score them vs current 1 question.
- **Exact IG formula for video captions**: Information gain calc assumes discrete answer space. For open-text clarifications, how to compute P(candidate matches answer) without fine-tuned classifier? Could use LLM or lexical overlap, but not validated for this repo's caption payloads.
- **"Discriminative facet" extraction from captions**: Sekulic et al. extract facets from *ranked docs*. Video captions are shorter. Feasibility of auto-faceting caption sets unknown—requires testing on actual VBS caption structure.
- **Multi-turn clarification**: Current repo fires ONE clarification. Literature (ClariQ, MIMICS-Click) supports multi-turn (ask follow-up to answer). Worth future work but likely exceeds VBS time budget.
- **Operator's verbal relay accuracy**: VBS moderator interprets operator's question relay and answers. No study on whether open-text clarification questions survive verbal relay better than structured MC (operationally relevant but outside IR literature scope).

---

## Summary Recommendations (for planner)

1. **Immediate (High-confidence, low latency)**: Adopt **facet-driven prompting** (Sekulic zero-shot). Extract differentiating attributes from top-5 candidate captions. Prefix LLM prompt with facet constraint. Implement in `query_processor.py:generate_clarification_question()`.
2. **Medium-term (Medium confidence, moderate complexity)**: Add **information-gain question selection**: generate 3-5 candidate questions, score by entropy reduction over candidate set, pick top-1. Integrate into `hybrid_search.py:compute_ambiguity_score()`. Requires lightweight scoring function—no training.
3. **Answer handling (High-confidence, low latency)**: Implement **in-place re-ranking** on clarification answer. Parse answer for entities/keywords; boost candidates matching keywords; re-sort. Implement in `hybrid_search.py` as new function.
4. **Avoid (for VBS)**: MCMIPL, full ICR multi-turn, finetuning-based approaches (ClarQ, ClariQ Stage 2 generation). All add latency or require training pipeline.


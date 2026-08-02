# VBS 2027 & Lightweight Hierarchical Indexing Research

## PART 1: VBS 2027 Specifics

### Findings

**No new concrete announcements beyond what you already know.** Organizers have NOT published:
- Rule changes or new task type previews
- Updated submission format specifications
- Specific registration deadline (only urging teams: "contact us as soon as possible")
- DRES API version 2026+ changes (only know v2.0.0 introduced POST submission endpoint in prior release)

**Confirmed Details:**
- **Event:** 16th edition VBS, January 5-8, 2027 (MMM 2027 co-location confirmed)
- **Venue:** Siem Reap, Cambodia
- **Dataset:** ~3,800 hours across V3C, MVK, LHE (as you stated)
- **Registration:** Contact organizers directly; no published deadline yet

**Key Reference:** Official VBS site states new teams should contact them early to "share data, hints, and maybe even some tools."

---

## PART 2: Lightweight Hierarchical Video Indexing

**Goal:** Achieve shot/scene-level retrieval benefits without H-EAGLE's full 3-level pipeline overhead.

### Option 1: Shot-Level Embedding Aggregation (Mean/Max Pooling)

**Source:** CVPR 2025 "Effectiveness of Max-Pooling for Fine-Tuning CLIP on Videos"; Video CLIP temporal research stream

**Core Idea:** Compute aggregated vectors at shot level by pooling frame embeddings already in Qdrant. Instead of querying frame-by-frame, first retrieve at coarse shot level via max-pooling or mean-pooling of constituent frame vectors.

**Implementation:**
- Store shot-to-frame mappings alongside Qdrant payloads
- Pre-compute aggregated vectors (max-pool or mean-pool of frame embeddings) at indexing time
- Build secondary index or derive shot vectors on-the-fly via vector aggregation

**Feasibility:** VERY HIGH. No new model, no retraining. Purely post-processing existing embeddings.

**Rough Effort:** 
- Pre-compute aggregations: 1-2 days (single pass over Qdrant + shot boundaries)
- Modify search pipeline to support two-level queries: 2-3 days
- Total: ~4-5 days

**Caveats:** Max-pooling works better for action/dynamic queries; mean-pooling is more general baseline. Research confirms mean-pooling "remains remarkably strong and computationally efficient" despite newer sophistication.

---

### Option 2: Lightweight Coarse Pre-Filter with Metadata Tags

**Source:** Multi-modal Multi-Tagger framework (Springer Nature 2025); Hierarchical Indexing with Knowledge Enrichment (arXiv:2510.09553)

**Core Idea:** 
- Extract discrete semantic tags (scene type, objects, actions) at shot level using lightweight tagging
- Build inverted index on tags (not vectors)
- Pre-filter by tags before fine-grained embedding search

**Implementation:**
- Use existing CLIP or lightweight vision-language model to tag sampled frames per shot (e.g., 1 tag per shot = negligible cost)
- Maintain tag → shot_ids mapping in memory/database
- At query time: extract query tags, retrieve candidate shots, then full embedding search

**Feasibility:** MEDIUM-HIGH. Requires tagging infrastructure but no retraining. Tags can be extracted once at indexing time.

**Rough Effort:**
- Design tag vocabulary + tagging strategy: 3-5 days
- Tag all shots in dataset: 2-3 days (amortized, done once)
- Integrate tag-based pre-filtering: 3-4 days
- Total: ~8-12 days

**Benefit:** Orders of magnitude faster coarse filtering if tag vocabulary is well-designed.

---

### Option 3: ProCLIP Two-Stage Framework (Prompt-Aware Lightweight Filtering)

**Source:** "Prompt-aware of Frame Sampling for Efficient Text-Video Retrieval" (arXiv:2507.15491)

**Core Idea:** 
Combine rapid coarse filtering (lightweight, query-aware) with fine-grained CLIP re-ranking. ProCLIP demonstrated **75% latency reduction** while maintaining accuracy (R@1=49.0 on MSR-VTT).

**Implementation:**
- Stage 1: Lightweight query-aware scorer (learned on small labeled set or heuristic) pre-selects candidate frames/shots
- Stage 2: Full CLIP fine-ranking on candidates

**Feasibility:** HIGH. The lightweight scorer can be a simple MLP trained on CLIP embeddings (no new video model needed). Works with existing frame embeddings directly.

**Rough Effort:**
- Design lightweight scorer architecture: 2-3 days
- Collect/annotate small training set (for learning scorer or use heuristic rules): 3-5 days
- Integrate two-stage pipeline: 2-3 days
- Total: ~7-11 days

**Key Advantage:** Query-awareness (adapts per query) vs. static shot aggregation. Proven 75% speedup is significant.

---

### Option 4: Cheap Shot Boundary Detection + Fixed-Interval Sampling

**Source:** "Scene Detection Policies and Keyframe Extraction Strategies for Large-Scale Video Analysis" (arXiv:2506.00667); TransNetV2 research

**Core Idea:** 
Apply lightweight shot detection (PySceneDetect: CPU-only, efficient) to define shot boundaries. Within each shot, apply fixed-interval sampling (e.g., every N frames) rather than adaptive sampling. Reduces redundant per-frame embeddings.

**Implementation:**
- Run PySceneDetect once per video (one-time cost, CPU-only)
- Store shot boundaries in payload
- At indexing time: only embed selected keyframes per shot (not every frame)
- Rebuild Qdrant with sparser, shot-aware embeddings

**Feasibility:** MEDIUM (requires re-indexing frames, which is costly). But post-hoc if current index already exists: HIGH (just run detector, update payload metadata).

**Rough Effort:**
- Run PySceneDetect on full dataset: 1-2 days (parallel-friendly)
- Update Qdrant payloads with shot metadata: 1 day
- If re-indexing needed: 3-5 additional days
- Total (post-hoc): ~3 days; (full re-index): ~6-7 days

**Trade-off:** Reduces embedding count (storage/search cost) but loses frame-level granularity. Useful if storage is bottleneck.

---

## Recommended Approach (Minimal Effort, Maximum Benefit)

**Combination of Options 1 + 3:**

1. **Immediately (1 week):** Implement shot-level aggregation (Option 1)
   - Pre-compute mean & max-pooled vectors for each shot
   - Add shot metadata to Qdrant payloads
   - Modify search to optionally return shot-level candidates

2. **Phase 2 (2-3 weeks, if needed):** Overlay ProCLIP lightweight filtering (Option 3)
   - Build lightweight scorer on top of aggregated embeddings
   - Two-stage search: coarse shots → fine frames

**Why:** Option 1 is almost free (post-hoc, no retraining). Option 3 proven effective in literature (75% speedup). Together: hierarchical benefit with minimal overhead.

---

## Summary Table

| Approach | Source | Core Tech | Feasibility | Effort | Benefit |
|----------|--------|-----------|-------------|--------|---------|
| Shot Aggregation | CVPR2025 | Mean/Max pool | Very High | 4-5d | Moderate (coarse retrieval) |
| Metadata Tags | 2025 papers | Inverted index | Medium-High | 8-12d | High (fast pre-filter) |
| ProCLIP | arXiv:2507 | Lightweight scorer | High | 7-11d | Very High (75% speedup) |
| Cheap Shot Detect | arXiv:2506 | PySceneDetect | Medium | 3-7d | Moderate (storage savings) |

---

## Unresolved Questions

1. **VBS 2027 Submission Constraints:** Will VBS 2027 enforce latency SLAs (e.g., per-query response time limits)? If yes, latency-reduction approaches (ProCLIP) become mandatory. **Action:** Contact organizers.

2. **Dataset Size at Competition:** Will the actual ~3,800-hour dataset be available pre-competition for indexing, or only at event time? Affects whether pre-computing aggregations is viable. **Action:** Check VBS website or contact organizers.

3. **DRES 2027 API Stability:** Will DRES use v2.0.0+ POST endpoint, or introduce breaking changes? Affects integration planning. **Action:** Monitor DRES GitHub releases; contact maintainers if unclear.

4. **Embedding Model Choice:** Are you locked into a specific CLIP version (ViT-L, ViT-B) or can you experiment? Aggregation strategies may differ by model dimension/capacity. **Action:** Document your embedding model choice.

5. **Baseline Performance:** Do you have a current frame-level search baseline (R@10, latency) to measure improvement against? Shot-level aggregation's benefit depends on query-to-shot variance.

---

## Sources

- [Video Browser Showdown Official Site](https://videobrowsershowdown.org/)
- [MMM 2027 Conference](https://www.mmm2027.net/)
- [DRES Project GitHub](https://github.com/dres-dev/DRES)
- [ProCLIP: Prompt-aware Frame Sampling](https://arxiv.org/pdf/2507.15491)
- [CVPR 2025: Max-Pooling for CLIP on Videos](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/html/Zohra_Effectiveness_of_Max-Pooling_for_Fine-Tuning_CLIP_on_Videos_CVPRW_2025_paper.html)
- [Scene Detection & Keyframe Extraction Strategies](https://arxiv.org/abs/2506.00667)
- [Hierarchical Indexing with Knowledge Enrichment](https://arxiv.org/pdf/2510.09553)
- [Multi-Modal Multi-Tagger Pre-screening](https://link.springer.com/article/10.1007/s44267-025-00073-2)
- [Video CLIP: Temporal & Fusion Strategies](https://www.emergentmind.com/topics/video-clip)
- [CLIP4Clip Aggregation Design Study](https://www.sciencedirect.com/science/article/abs/pii/S0925231224006763)

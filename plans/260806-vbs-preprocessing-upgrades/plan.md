---
title: "VBS preprocessing upgrades from NOTE.md"
date: 2026-08-06
status: completed
project: tgltw-vbs-2027
---

# VBS preprocessing upgrades from NOTE.md

## Context

The attached note proposes improvements to the offline VBS preprocessing
pipeline. The repository is currently on `main` after the Vietnamese-specific
OCR path was removed. The implementation must stay inside `preprocessing/`
unless a shared model import or test fixture requires a small adjacent change.

The VBS/V3C collection already publishes shot boundaries, representative
keyframes, metadata, analysis data, and ASR. Reusing those assets is safer and
faster than recomputing equivalent signals. The local repository does not
contain a real V3C asset sample, so all asset parsing must be optional,
schema-tolerant, and covered by synthetic fixtures.

## Decisions from the note review

### Implement now

1. Optional V3C asset adapter:
   - detect `msb/`, `keyframes/`, `metadata/`, and `asr/` below a configured
     asset root;
   - consume official shot boundaries instead of PySceneDetect when a valid
     file is present;
   - consume supplied ASR and metadata instead of running duplicate local
     analysis when matching files are present;
   - expose official keyframe paths and use them only when the file-to-shot
     mapping is unambiguous, otherwise fall back to raw-video extraction.
2. Batch Qdrant writes for visual and ambient-audio points, with an explicit
   flush at video boundaries and process completion.
3. Quality-aware keyframe selection using a cheap Laplacian sharpness score as
   a small tie-break/bonus on top of existing semantic farthest-point
   sampling. This is CPU-only and does not add another model.
4. Tests for asset parsing, sharpness scoring, selection behavior, and batched
   indexer flushing.
5. Documentation and `.env.template` entries explaining opt-in assets,
   fallback behavior, and the new batch/quality settings.

### Keep existing but do not force on

- SigLIP secondary embeddings already exist and remain opt-in. Enabling them
  changes the Qdrant schema and requires a full re-index, so this change does
  not silently turn them on.
- PP-OCRv6 English/multilingual behavior remains as currently configured. The
  Vietnamese accent-map/Vintern path is already removed; the note's old OCR
  fix is therefore obsolete.

### Defer explicitly

- TransNetV2 replacement: useful for a controlled benchmark, but the package
  and model-serving contract are not present in this repository. Replacing
  PySceneDetect without a real V3C sample risks changing shot/frame mapping.
- H-EAGLE three-level semantic indexing: requires new shot/action index
  contracts and retrieval changes, outside this preprocessing-only task.
- PraK localized/multizone embeddings: high implementation and query/UI cost;
  the note correctly flags it as hard and it is not required for baseline
  preprocessing.
- Emotion indexing: adds another noisy model and storage dimension without a
  validated VBS query need.
- Full SAM/learned saliency over every candidate and ffmpeg I-frame-only
  extraction: both need corpus-level recall benchmarks before enabling them.

## Phase 1 — Codebase and fixture contracts

### Goal

Define stable, testable interfaces around V3C assets and current sampling.

### Tasks

- Add a `V3CAssetStore` module with strict path discovery and tolerant parsers
  for common V3C formats:
  - whitespace/tab-separated master-shot rows;
  - per-video JSON metadata;
  - per-video ASR CSV rows `(start, end, transcript)`;
  - per-video representative keyframe images.
- Preserve shot IDs, source paths, timestamps, and optional frame indices in a
  normalized internal record.
- Add synthetic fixtures covering valid, missing, malformed, and ambiguous
  asset files.
- Add configuration for asset root, enable/disable behavior, official
  keyframe usage, and raw-video fallback.

## Phase 2 — Integrate safe offline optimizations

### Goal

Use valid official assets and reduce indexing overhead without changing the
current behavior when assets are absent.

### Tasks

- In `preprocessing/main.py`, initialize the asset store once and select
  official scenes/ASR/metadata per video only when a matching asset is valid.
- Keep local ASR and PySceneDetect as explicit fallbacks, with log messages
  showing which source was selected.
- Use official keyframes only for unambiguous matches; include `shot_id` and
  `asset_source` in payloads.
- Add Laplacian quality scoring to adaptive selection with a conservative,
  configurable weight and no additional model call.
- Buffer Qdrant writes and flush after each video plus at shutdown.

## Phase 3 — Documentation and operational safety

### Goal

Make the changes understandable and safe to deploy on the VBS server.

### Tasks

- Document expected V3C asset layout, example environment variables, and
  fallback semantics in preprocessing README/config templates.
- Document deferred items and why they are not enabled by default.
- Add clear logs for official-asset hit/miss counts and batch flushes.

## Phase 4 — Verification

### Goal

Prove behavior with unit tests and a lightweight smoke run without requiring
GPU models or a full V3C download.

### Tasks

- Run focused preprocessing tests with fake model/indexer objects.
- Run the complete existing test suite.
- Run syntax/import checks for the preprocessing package.
- Run a small synthetic end-to-end path that verifies:
  - official scene/ASR/metadata selection;
  - raw fallback when an asset is missing or malformed;
  - no point loss across batch flushes;
  - quality score is finite and selection remains bounded by budget.
- Record remaining caveats and deployment settings in the final handoff.

## Acceptance criteria

- An ordinary dataset with no V3C asset directories behaves as before except
  for batched Qdrant writes and the small configurable sharpness bonus.
- A synthetic V3C layout is parsed correctly and never causes a crash when one
  asset family is absent or malformed.
- Qdrant receives points in batches and all buffered points are flushed before
  the process exits.
- No PraK/H-EAGLE/emotion/TransNet dependency is added to the default runtime.
- Focused and full tests pass.

## Research basis

- Official VBS existing-data page: V3C publishes shot boundaries, keyframes,
  metadata/analysis, ASR, and points teams to TransNetV2.
- V3C/NIST dataset descriptions: one representative keyframe per shot and
  per-video metadata are part of the collection.
- TransNetV2 paper/repository: strong shot-boundary model, but not integrated
  here without a reproducible local serving contract and benchmark.
- Qdrant collection/points documentation: collections support named vectors
  and batched point uploads; the implementation uses batching without changing
  the existing schema.

---
title: "Migrate ASR from PhoWhisper to faster-whisper large-v3-turbo"
description: "Swap PhoWhisper/transformers ASR for faster-whisper (CTranslate2) Whisper large-v3-turbo, plus VAD filtering, confidence-based hallucination filtering, and word-level timestamps."
status: pending
priority: P1
effort: 2.5h
branch: main
tags: [asr, preprocessing, faster-whisper, whisper-turbo, hallucination-filter, vbs2027]
created: 2026-08-04
---

# ASR Migration: PhoWhisper → faster-whisper Whisper large-v3-turbo

## Why

1. **Wrong model for the dataset.** `vinai/PhoWhisper-large` is Vietnamese-specialised; the VBS dataset (V3C) is not Vietnamese-centric like the sibling AIC dataset. Whisper large-v3-turbo is 99-language, MIT-licensed.
2. **Wrong library.** `transformers.pipeline("automatic-speech-recognition")` hides the per-segment signals we need. faster-whisper (CTranslate2) is ~4x faster, uses less VRAM, and exposes `avg_logprob` / `no_speech_prob` / `compression_ratio` / `words` plus a built-in silero VAD filter. 3 VBS-2026 teams did the same swap (`vbs-2026-paper-methods-and-team-analysis.md` L108, L195, L778).
3. **Live bug.** `preprocessing/main.py:165-191` and `:421-442` embed + index **every** ASR segment with zero filtering — Whisper's classic hallucination-over-silence output goes straight into the dense index as junk speech points.
4. **Coarse timestamps.** Current output is 30s-chunk granularity; word-level timestamps give the VQA verification pipeline real hotspot localisation ("Ưu tiên 3 — VQA guided verification", step 2).

## Phases

| # | Phase | Status | Effort |
|---|-------|--------|--------|
| 01 | [faster-whisper turbo ASR migration + 3 hardening techniques](./phase-01-faster-whisper-turbo-asr-migration.md) | pending | 2.5h |

One cohesive change = one PR (project convention: 1 phase = 1 PR).

## Key decisions (already confirmed, do not re-litigate)

- Model: **Whisper large-v3-turbo**, CT2 repo `deepdml/faster-whisper-large-v3-turbo-ct2` (MIT, CT2-only files: `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.json`, `preprocessor_config.json` — no duplicate safetensors bulk).
- Library: **faster-whisper >= 1.1.0** (bundled silero VAD, `turbo` support).
- Hardening: (a) `vad_filter=True`, (b) confidence/hallucination post-filter before embed+index, (c) `word_timestamps=True`.

## Key dependencies / blast radius

- `models/asr.py` (rewrite, class rename `PhoWhisperASR` → `WhisperASR`)
- `preprocessing/audio/asr_segment_filter.py` (new, pure function — the only new file)
- `preprocessing/audio/audio_processor.py`, `preprocessing/main.py` (2 call sites)
- `preprocessing/config.py:46`, `preprocessing/.env.template:27`, `download_assets.py:278,289-290`
- `preprocessing/requirements.txt`, `pyproject.toml` (`preprocessing` group) + `uv lock`
- `README.md:32,78,114,307`; `CHANGELOG.md` + `preprocessing/CHANGELOG.md`
- No `docs/` dir in this repo — docs impact is README + 2 changelogs only.

## Reindex note

Speech points already in Qdrant were produced by PhoWhisper and are unfiltered. Payload schema is additive (new `timestamp_end`, `words`, `asr_avg_logprob` keys) so nothing breaks, but the audio stage must be re-run to gain the new quality. Coordinate with the team before touching the shared collection.

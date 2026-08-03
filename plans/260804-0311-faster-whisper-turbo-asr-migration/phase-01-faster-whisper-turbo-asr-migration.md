# Phase 01 — faster-whisper Whisper large-v3-turbo ASR + 3 hardening techniques

## Context Links

- Overview: [plan.md](./plan.md)
- Method survey (motivation): `vbs-2026-paper-methods-and-team-analysis.md` L108 (Fusionista2.0 "ASR: faster-whisper thay Whisper"), L195 (VIREO-style "Whisper-Turbo"), L778 ("Whisper/faster-whisper: speech"), "Ưu tiên 3 — VQA guided verification" step 2 (hotspot from transcript)
- Current code: `models/asr.py` (62 lines), `preprocessing/audio/audio_processor.py` (79 lines), `preprocessing/main.py:165-191` + `:421-442`
- Prior ffmpeg-PATH workaround history: `preprocessing/CHANGELOG.md:45`
- Upstream refs: <https://github.com/SYSTRAN/faster-whisper>, <https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2>

## Overview

- **Priority:** P1
- **Status:** pending
- **Effort:** ~2.5h
- Replace the PhoWhisper/transformers ASR wrapper with a faster-whisper (CTranslate2) Whisper large-v3-turbo wrapper, and stop indexing hallucinated/silent segments.

## Key Insights (verified this session)

1. **Segment fields exist and are exactly what we need.** `faster_whisper.transcribe.Segment` is a dataclass with `id, seek, start, end, text, tokens, avg_logprob, compression_ratio, no_speech_prob, words, temperature`; `Word` has `start, end, word, probability`. Verified against upstream `faster_whisper/transcribe.py`.
2. **faster-whisper does NOT drop hallucinated segments for us.** Its internal `log_prob_threshold`/`no_speech_threshold` only drive temperature fallback and whole-window skipping; low-confidence segments still appear in the output iterator. An explicit post-filter is required. Use OpenAI Whisper's own reference thresholds as defaults: `logprob < -1.0`, `no_speech_prob > 0.6`, `compression_ratio > 2.4`.
3. **No ffmpeg binary needed by faster-whisper.** Audio is decoded with PyAV, which bundles the FFmpeg *libraries* in its wheel. So the `models/asr.py:7-14` PATH shim (added purely because `transformers`' ASR pipeline shells out to a bare `ffmpeg`) becomes dead code. `AudioProcessor.extract_audio` (`audio_processor.py:29-37`) has its own independent `bin/ffmpeg` fallback and does **not** depend on the shim's global `PATH` mutation — grep confirms these are the only two ffmpeg touchpoints. → **remove the shim**, document the removal in `preprocessing/CHANGELOG.md`.
4. **The already-extracted 16kHz mono WAV can be passed straight through** as a path — PyAV handles WAV natively. No need to pre-decode to numpy.
5. **CTranslate2 has no MPS backend.** Devices are `cuda` / `cpu` only. Current `asr.py:26` picks `mps` on Apple Silicon — that must be dropped or CT2 raises. Use `ctranslate2.get_cuda_device_count()` (already a faster-whisper dep) instead of `torch.cuda.is_available()`, which also drops the `torch` import from `asr.py`.
6. **Repo id choice.** faster-whisper's own shorthand maps `"large-v3-turbo"`/`"turbo"` → `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (also MIT, fp16). We must configure an **explicit repo id** anyway because `download_assets.py` runs `hf download <repo_id>` and shorthands aren't valid repo ids. Chosen: `deepdml/faster-whisper-large-v3-turbo-ct2` — MIT per model card, and its file list contains only CT2 artifacts (no redundant `.safetensors`), so the download stays ~1.6GB.
7. **`end_t` is currently dead.** `main.py:176` and `:431` compute `end_t = seg["end"]` and never use it — the speech payload only stores `timestamp` (start). Fixing this is free while we're here.

## Requirements

### Functional

- FR1: ASR runs on faster-whisper with Whisper large-v3-turbo, multilingual auto-detect by default.
- FR2: `vad_filter=True` (silero) skips non-speech regions.
- FR3: Segments failing confidence/hallucination checks are dropped **before** `embed_text()` + `indexer.index_visual_point()` at both `main.py` call sites.
- FR4: `word_timestamps=True`; per-word `{word, start, end}` reaches the Qdrant payload.
- FR5: The `{"text","start","end"}` dict contract of `AudioProcessor.transcribe_audio()` is preserved (additive keys only) so nothing downstream breaks.
- FR6: All thresholds and flags are env-tunable via `preprocessing/config.py`.

### Non-functional

- `models/asr.py` stays well under 200 lines (target ~110).
- New filter logic is a pure function, unit-testable with stub dicts, **no model download / no torch import**.
- try/except around model load and around transcription (the segment iterator is lazy — errors surface during iteration, not at the `transcribe()` call).
- Local `weights/` dir takes precedence over the Hub, matching the existing pattern (`asr.py:21-24`).

## Architecture

```
video/audio file
   └─ AudioProcessor.extract_audio()      (ffmpeg → 16kHz mono WAV, unchanged)
        └─ AudioProcessor.transcribe_audio()
             ├─ WhisperASR.transcribe(wav_path)        models/asr.py
             │    └─ faster_whisper.WhisperModel.transcribe(
             │           vad_filter=True,              ← hardening #1 (silero VAD)
             │           word_timestamps=True)         ← hardening #3
             │       → dicts: text/start/end/avg_logprob/no_speech_prob/
             │                compression_ratio/words/language
             └─ filter_asr_segments(...)   preprocessing/audio/asr_segment_filter.py
                                                       ← hardening #2 (confidence filter)
                  └─ main.py L165-191 / L421-442 : embed_text() → index_visual_point()
```

**Single filter application point** (`transcribe_audio`) covers both `main.py` call sites — DRY, and both sites consume only filtered output. The filter itself stays a pure module-level function so it is testable in isolation.

## Related Code Files

**Modify**

| File | Change |
|------|--------|
| `models/asr.py` | Full rewrite of the class body; `PhoWhisperASR` → `WhisperASR`; drop `torch` + ffmpeg-PATH shim |
| `preprocessing/audio/audio_processor.py` | L10 import, L19 instantiation, L55-61 `transcribe_audio` applies filter + logs drop count; docstring L57 |
| `preprocessing/main.py` | L173-191 and L428-442: add `timestamp_end` + `words` (+ `asr_avg_logprob`) to the speech payload, use the previously-dead `end_t` |
| `preprocessing/config.py` | L46: `PHOWHISPER_MODEL_ID` → `ASR_MODEL_ID` + 6 new `ASR_*` vars |
| `preprocessing/.env.template` | L27: same rename/default + new vars with comments |
| `download_assets.py` | L278 var+key+default, L289-290 comment + call |
| `preprocessing/requirements.txt` | add `faster-whisper>=1.1.0` |
| `pyproject.toml` | add `faster-whisper>=1.1.0` to the `preprocessing` dependency group, then `uv lock` |
| `README.md` | L32, L78, L114, L307 |
| `CHANGELOG.md` | new `[1.16.0]` entry |
| `preprocessing/CHANGELOG.md` | new `[1.5.0]` entry (incl. the shim removal, superseding the L45 note) |

**Create**

- `preprocessing/audio/asr_segment_filter.py` (~45 lines, pure)
- `tests/test_asr_segment_filter.py` (first test file in the repo — see step 9)

**Delete**

- Nothing. (`weights/PhoWhisper-large/` ~3GB can be removed manually by whoever already downloaded it — mention in the PR description, don't script it.)

## Implementation Steps

1. **`preprocessing/config.py`** — replace L46 with the ASR block:
   ```python
   # ASR (faster-whisper / CTranslate2). Whisper large-v3-turbo: 99-language,
   # MIT. Must be a CT2-converted repo id (download_assets.py runs `hf download`
   # on it, so faster-whisper's "turbo" shorthand is not usable here).
   ASR_MODEL_ID = os.getenv("ASR_MODEL_ID", "deepdml/faster-whisper-large-v3-turbo-ct2")
   # Blank = auto-detect per file (V3C is multilingual). Set e.g. "en" to force.
   ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "") or None
   # CTranslate2 compute type: "auto" picks float16 on CUDA / int8 on CPU.
   ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "auto")
   ASR_VAD_FILTER_ENABLED = os.getenv("ASR_VAD_FILTER_ENABLED", "true").lower() == "true"
   ASR_WORD_TIMESTAMPS_ENABLED = os.getenv("ASR_WORD_TIMESTAMPS_ENABLED", "true").lower() == "true"
   # Hallucination/silence filter thresholds - OpenAI Whisper's own reference
   # values. faster-whisper reports these per segment but does NOT drop the
   # segment for us, so we filter before embedding+indexing.
   ASR_MIN_AVG_LOGPROB = float(os.getenv("ASR_MIN_AVG_LOGPROB", -1.0))
   ASR_MAX_NO_SPEECH_PROB = float(os.getenv("ASR_MAX_NO_SPEECH_PROB", 0.6))
   ASR_MAX_COMPRESSION_RATIO = float(os.getenv("ASR_MAX_COMPRESSION_RATIO", 2.4))
   ```
2. **`preprocessing/.env.template`** — replace L27 with `ASR_MODEL_ID=deepdml/faster-whisper-large-v3-turbo-ct2` plus the new vars (`ASR_LANGUAGE=` blank, `ASR_COMPUTE_TYPE=auto`, `ASR_VAD_FILTER_ENABLED=true`, `ASR_WORD_TIMESTAMPS_ENABLED=true`, `ASR_MIN_AVG_LOGPROB=-1.0`, `ASR_MAX_NO_SPEECH_PROB=0.6`, `ASR_MAX_COMPRESSION_RATIO=2.4`) with one-line comments matching config.py's wording.
3. **`models/asr.py`** — rewrite. Keep the bare `from config import ...` style already used at L5 (pre-existing repo convention; do not "fix" it here). Shape:
   ```python
   import os
   from typing import List, Dict, Any
   from config import (ASR_MODEL_ID, ASR_LANGUAGE, ASR_COMPUTE_TYPE,
                       ASR_VAD_FILTER_ENABLED, ASR_WORD_TIMESTAMPS_ENABLED)

   class WhisperASR:
       """ASR wrapping faster-whisper (CTranslate2) Whisper large-v3-turbo."""
       def __init__(self, model_id: str = ASR_MODEL_ID):
           local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "weights", model_id.split("/")[-1])
           if os.path.exists(local_path):
               model_id = local_path
           # CTranslate2 supports cuda/cpu only - no MPS backend.
           self.device = "cuda" if _cuda_available() else "cpu"
           ...
           from faster_whisper import WhisperModel
           self.model = WhisperModel(model_id, device=self.device,
                                     compute_type=ASR_COMPUTE_TYPE)
   ```
   - `_cuda_available()`: `try: import ctranslate2; return ctranslate2.get_cuda_device_count() > 0 except Exception: return False`.
   - Wrap the `WhisperModel(...)` construction in try/except and re-raise with a message naming the model id (missing/incomplete local weights dir is the likely failure).
   - `transcribe(audio_path)`: call `self.model.transcribe(audio_path, language=ASR_LANGUAGE, beam_size=5, vad_filter=ASR_VAD_FILTER_ENABLED, word_timestamps=ASR_WORD_TIMESTAMPS_ENABLED)`. **Iterate the returned generator inside the try block** — it is lazy, so decode errors raise during iteration. On exception, print and `return []` (matches the pipeline's existing "silent video → empty list" tolerance at `audio_processor.py:50-53`).
   - Per segment emit:
     ```python
     {"text": seg.text.strip(), "start": float(seg.start), "end": float(seg.end),
      "avg_logprob": float(seg.avg_logprob),
      "no_speech_prob": float(seg.no_speech_prob),
      "compression_ratio": float(seg.compression_ratio),
      "words": [{"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in (seg.words or [])],
      "language": info.language}
     ```
     `text`/`start`/`end` keys are unchanged from the old wrapper → contract preserved. Drop `probability` per word (payload bulk over ~3800h; the segment-level `avg_logprob` is the signal we act on).
   - Delete the `import torch`, the `mps` branch, and the L7-14 ffmpeg-PATH shim.
4. **`preprocessing/audio/asr_segment_filter.py`** — new pure module:
   ```python
   def filter_asr_segments(segments, min_avg_logprob=ASR_MIN_AVG_LOGPROB,
                           max_no_speech_prob=ASR_MAX_NO_SPEECH_PROB,
                           max_compression_ratio=ASR_MAX_COMPRESSION_RATIO,
                           min_chars=2) -> List[Dict[str, Any]]:
   ```
   Drop a segment when: stripped `text` is empty or shorter than `min_chars`; `no_speech_prob > max_no_speech_prob`; `avg_logprob < min_avg_logprob`; `compression_ratio > max_compression_ratio`. **Missing keys → treat as passing** (permissive), so a future backend that doesn't report confidences never silently drops everything. No printing inside the function (keep it pure); thresholds as keyword args so tests override explicitly instead of monkeypatching config.
5. **`preprocessing/audio/audio_processor.py`** — L10 `from models.asr import WhisperASR`; L19 `self.asr_model = WhisperASR()`; L55-61:
   ```python
   raw = self.asr_model.transcribe(audio_path)
   kept = filter_asr_segments(raw)
   if len(kept) != len(raw):
       print(f"ASR filter: dropped {len(raw) - len(kept)}/{len(raw)} low-confidence/silent segments.")
   return kept
   ```
   Update the docstring (L57) — no longer "PhoWhisper", and state that filtering happens here.
6. **`preprocessing/main.py` site A (L173-191)** — extend the payload with `"timestamp_end": end_t` and `"words": seg.get("words", [])` and `"asr_avg_logprob": seg.get("avg_logprob")`. `end_t` (L176) stops being dead. `print` at L170 now reports the post-filter count — that is the desired behaviour, no change needed.
7. **`preprocessing/main.py` site B (L428-442)** — identical payload additions for standalone audio files.
8. **`download_assets.py`** — L278 → `asr_model_id = env_vars.get("ASR_MODEL_ID", "deepdml/faster-whisper-large-v3-turbo-ct2")`; L289 comment → `# 1. Whisper large-v3-turbo (CTranslate2 format, for faster-whisper)`; L290 → `download_model(asr_model_id, asr_model_id.split("/")[-1], token=hf_token)`. `download_model()` (L55-72) is already a generic `hf download` wrapper — no change needed to it. Resulting dir `weights/faster-whisper-large-v3-turbo-ct2/` is exactly what `WhisperModel(local_dir)` expects.
9. **Deps** — `preprocessing/requirements.txt`: add `faster-whisper>=1.1.0` with a comment noting it pulls `ctranslate2`, `av` (bundles FFmpeg libs → no ffmpeg binary needed) and `onnxruntime` (bundled silero VAD), and that CT2 is **CUDA-or-CPU only, no Apple MPS**. `pyproject.toml`: add the same to the `preprocessing` dependency group (not `inference` — `inference-code/` and `webapp/` never touch ASR; grep confirms `models/asr.py`'s only importer is `preprocessing/audio/audio_processor.py`). Run `uv lock`.
10. **Tests** — `tests/test_asr_segment_filter.py`. This repo currently has **no `tests/` dir and no pytest dep**, so write it to run both under pytest and as a plain script (`if __name__ == "__main__":` calling the test functions). Cases: keeps a good segment; drops empty/whitespace text; drops `no_speech_prob=0.95`; drops `avg_logprob=-2.5`; drops `compression_ratio=5.0`; keeps a segment with confidence keys absent (permissive path); returns `[]` for `[]`. Stub dicts only — no model, no torch, no network.
11. **Compile check** — `python -m py_compile models/asr.py preprocessing/audio/asr_segment_filter.py preprocessing/audio/audio_processor.py preprocessing/main.py preprocessing/config.py download_assets.py`.
12. **Docs** — `README.md` L32 (`asr.py` comment → faster-whisper Whisper large-v3-turbo transcriber), L78 (`WhisperASR`, mention VAD + confidence filter + word timestamps), L114 (download list wording), L307 (third-party weights list: replace PhoWhisper with "Whisper large-v3-turbo (MIT, CTranslate2 conversion)"). Add `CHANGELOG.md` `[1.16.0]` and `preprocessing/CHANGELOG.md` `[1.5.0]` entries; the preprocessing entry must explicitly note that the L45 ffmpeg-PATH workaround is removed and why it is no longer needed.

## Todo List

- [ ] 1. `preprocessing/config.py`: `ASR_MODEL_ID` + 6 new `ASR_*` vars
- [ ] 2. `preprocessing/.env.template`: matching vars/defaults/comments
- [ ] 3. `models/asr.py`: rewrite as `WhisperASR` (faster-whisper); drop torch, mps, ffmpeg shim
- [ ] 4. `preprocessing/audio/asr_segment_filter.py`: pure `filter_asr_segments()`
- [ ] 5. `audio_processor.py`: import/instantiate + apply filter + drop-count log + docstring
- [ ] 6. `main.py` site A (L173-191): payload gains `timestamp_end`, `words`, `asr_avg_logprob`
- [ ] 7. `main.py` site B (L428-442): same payload additions
- [ ] 8. `download_assets.py`: `ASR_MODEL_ID` rename + comment
- [ ] 9. `requirements.txt` + `pyproject.toml` (`preprocessing` group) + `uv lock`
- [ ] 10. `tests/test_asr_segment_filter.py` (pytest-or-script, stub dicts)
- [ ] 11. `py_compile` all touched files; run the filter test
- [ ] 12. README (L32/78/114/307) + both CHANGELOGs
- [ ] 13. `code-reviewer` agent pass
- [ ] 14. PR → merge

## Success Criteria

- `grep -ri "phowhisper" --include="*.py" --include="*.md" .` returns hits only in CHANGELOG history entries.
- `python -m py_compile` clean on all touched files; `from models.asr import WhisperASR` imports without error once `faster-whisper` is installed.
- Filter unit tests pass with no network/model access.
- One real sample video processed end-to-end: speech point count is **lower** than a PhoWhisper run on the same file, no indexed segment has empty/repetitive text, and each speech payload carries a non-empty `words` array plus `timestamp_end > timestamp`.
- Console shows the `ASR filter: dropped N/M ...` line on a video with music/silence stretches.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| CTranslate2 has no MPS → Apple Silicon dev boxes fall back to int8 CPU and get slower ASR than the old MPS pipeline | Accepted: the competition/preprocessing host is CUDA. Document in `requirements.txt` + README so nobody debugs a phantom perf regression. |
| Thresholds too aggressive → real speech dropped | All three thresholds are env vars; drop counts are logged; validate on 2-3 sample V3C videos before a full run. Defaults are OpenAI's own reference values, not invented. |
| `language` auto-detect only inspects the first window → mislabels multilingual videos | `ASR_LANGUAGE` blank (auto) by default; can be pinned per dataset if V3C turns out effectively English-only. |
| Word arrays inflate Qdrant payload over ~3800h | Only `{word,start,end}` stored, rounded to 2 decimals, `probability` dropped; `ASR_WORD_TIMESTAMPS_ENABLED=false` is an escape hatch. |
| Removing the ffmpeg-PATH shim breaks something relying on its global side effect | Verified: only other ffmpeg use is `AudioProcessor.extract_audio`, which resolves `bin/ffmpeg` itself; librosa reads the pre-extracted WAV via soundfile. |
| First run silently downloads ~1.6GB from the Hub if `weights/` is empty | `download_assets.py` step is the documented path (README L114); local-weights check in `WhisperASR.__init__` short-circuits it afterwards. Public repo → no `HF_TOKEN` required. |
| Old unfiltered PhoWhisper speech points remain in the shared Qdrant collection | Payload change is additive so nothing breaks; flag re-preprocessing of the audio stage as a team decision in the PR, do not unilaterally wipe the shared collection. |

## Security Considerations

- No new runtime network calls beyond the one-time Hub model fetch; weights load from `weights/` after that. Public MIT repo, no token/secret needed.
- No new subprocess or shell invocation — the PATH-mutating shim is removed, and decoding moves in-process to PyAV. Net reduction in shell surface.
- Untrusted media decoding risk is unchanged in kind (PyAV/FFmpeg libs instead of the FFmpeg binary); keep the try/except so a malformed file yields `[]` instead of aborting a long batch run.
- `.env` still holds all overrides; no credentials added; nothing new to commit.

## Deferred / out of scope (YAGNI)

- Cross-segment duplicate-text collapse (Whisper's repeated "Thank you for watching" across consecutive segments). `compression_ratio` only catches *within*-segment repetition. Cheap to add later if sample runs show it; not one of the 3 confirmed techniques.
- `BatchedInferencePipeline` (faster-whisper batched mode) for extra throughput.
- Word-level payload consumption by the VQA hotspot verifier — this phase only *produces* the data.

## Next Steps

1. Implement steps 1-12 on a branch (`feat/faster-whisper-turbo-asr`, off `main`).
2. Run the filter unit test + `py_compile`; then a single-video smoke run against the success criteria.
3. `code-reviewer` agent pass; fix findings.
4. PR → merge.
5. **Follow-up (project convention, not part of this plan):** update the Notion page "Our method (VBS)" (page_id `297721d0cc698332a4ed81d0278b68bf`) audio/ASR section after the PR merges.
6. Team decision required before re-running preprocessing on the shared Qdrant collection (see reindex note in `plan.md`).

## Unresolved Questions

1. **Repo id preference:** default set to `deepdml/faster-whisper-large-v3-turbo-ct2` (MIT, CT2-only files). faster-whisper's own `"turbo"` shorthand instead resolves to `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (also MIT, fp16). Anyone typing `ASR_MODEL_ID=turbo` would get the mobiuslabs one via the Hub while `download_assets.py` would fail on the shorthand — accept, or normalise shorthands in `download_assets.py`?
2. **`tests/` dir:** this would be the repo's first test file and pytest is not a declared dependency. OK to add `tests/` (script-runnable, pytest optional), or keep the check inline as a `__main__` block in `asr_segment_filter.py`?
3. **`ASR_LANGUAGE` default:** left blank (auto-detect) for a 99-language model. If V3C is effectively English-only, pinning `en` is both faster and less hallucination-prone — worth confirming against the actual dataset.
4. **Whether to keep `probability` per word.** Dropped for payload size; needed if the VQA verifier later wants per-word confidence.

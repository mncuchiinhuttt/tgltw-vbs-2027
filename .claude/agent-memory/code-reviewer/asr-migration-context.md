---
name: asr-migration-context
description: PhoWhisper to faster-whisper (Whisper large-v3-turbo) ASR swap for VBS/V3C — rationale, files, and what's deliberately deferred
metadata:
  type: project
---

As of 2026-08-04 (CHANGELOG.md 1.16.0 / preprocessing/CHANGELOG.md 1.5.0),
`models/asr.py`'s `WhisperASR` wraps `faster_whisper.WhisperModel`
(CTranslate2) running Whisper large-v3-turbo (`deepdml/faster-whisper-large-v3-turbo-ct2`),
replacing the old `PhoWhisperASR` (transformers pipeline, `vinai/PhoWhisper-large`).

**Why:** this project (`tgltw-vbs-2027`) targets VBS/V3C, which is not
Vietnamese-centric — PhoWhisper (Vietnamese-specialized) was inherited from a
sibling AIC-2026 codebase where it made sense, but is the wrong fit here.

**How to apply:** if reviewing further ASR-adjacent changes, know that:
- CTranslate2 has no MPS backend — device selection is cuda/cpu only via
  `ctranslate2.get_cuda_device_count()`, never `torch.cuda.is_available()`/mps.
- faster-whisper does NOT drop hallucinated/silent segments itself; a
  dedicated post-filter (`preprocessing/audio/asr_segment_filter.py`,
  `filter_asr_segments()`) runs before embedding+indexing, using OpenAI
  Whisper's own reference thresholds (avg_logprob < -1.0, no_speech_prob >
  0.6, compression_ratio > 2.4) as defaults, permissive on missing keys.
- The old ffmpeg-PATH shim at the top of `models/asr.py` is gone — faster-whisper
  decodes via PyAV (bundles FFmpeg libs), no bare `ffmpeg` binary needed for
  ASR specifically (separate from `AudioProcessor.extract_audio`'s own
  independent `bin/ffmpeg` fallback for video-to-WAV extraction).
- Re-indexing the shared Qdrant collection with the new ASR output is an
  explicit deferred team decision, not bundled into the migration PR — old
  PhoWhisper speech points remain until that's decided (payload changes are
  additive so nothing breaks in the meantime).
- Full payload (`words` word-level timestamp array, `asr_avg_logprob`,
  `timestamp_end`) is additive to the speech Qdrant payload, and
  `webapp/backend/main.py` passes Qdrant payloads through to API responses
  verbatim (`"payload": hit["payload"]`) — so the `words` array now flows to
  the frontend on every speech hit; worth watching for response-size/latency
  impact at query time given VBS's real-time constraints.

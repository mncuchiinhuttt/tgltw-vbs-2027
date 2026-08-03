import os
from typing import List, Dict, Any
from config import (
    ASR_MODEL_ID, ASR_LANGUAGE, ASR_COMPUTE_TYPE,
    ASR_VAD_FILTER_ENABLED, ASR_WORD_TIMESTAMPS_ENABLED,
)


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class WhisperASR:
    """
    ASR (Speech-to-Text) module wrapping faster-whisper (CTranslate2),
    running Whisper large-v3-turbo by default.
    """
    def __init__(self, model_id: str = ASR_MODEL_ID):
        # Check if local weights path exists under global weights/
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

        # CTranslate2 supports cuda/cpu only - no Apple Silicon MPS backend.
        self.device = "cuda" if _cuda_available() else "cpu"
        print(f"Loading ASR model: {model_id} on {self.device}...")

        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_id, device=self.device, compute_type=ASR_COMPUTE_TYPE)
        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model '{model_id}': {e}") from e
        print("ASR model loaded successfully.")

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        segments = []
        try:
            segment_iter, info = self.model.transcribe(
                audio_path,
                language=ASR_LANGUAGE,
                beam_size=5,
                vad_filter=ASR_VAD_FILTER_ENABLED,
                word_timestamps=ASR_WORD_TIMESTAMPS_ENABLED,
            )
            # The segment iterator is lazy - decode/inference errors surface
            # here during iteration, not at the transcribe() call above.
            for seg in segment_iter:
                segments.append({
                    "text": seg.text.strip(),
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "avg_logprob": float(seg.avg_logprob),
                    "no_speech_prob": float(seg.no_speech_prob),
                    "compression_ratio": float(seg.compression_ratio),
                    "words": [
                        {"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                        for w in (seg.words or [])
                    ],
                    "language": info.language,
                })
        except Exception as e:
            print(f"ASR transcription failed for {audio_path}: {e}")
            return []
        return segments

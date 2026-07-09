import os
import shutil
import torch
from typing import List, Dict, Any
from config import PHOWHISPER_MODEL_ID

# transformers' ASR pipeline shells out to a bare "ffmpeg" command to decode
# audio files, with no way to point it at a specific binary. If ffmpeg isn't
# on PATH, fall back to the workspace's bundled bin/ffmpeg by prepending its
# directory to PATH so that internal call can still resolve it.
if shutil.which("ffmpeg") is None:
    _bin_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin")
    if os.path.exists(os.path.join(_bin_dir, "ffmpeg")):
        os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

class PhoWhisperASR:
    """
    ASR (Speech-to-Text) module wrapping PhoWhisper.
    """
    def __init__(self, model_id: str = PHOWHISPER_MODEL_ID):
        # Check if local weights path exists under global weights/
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading ASR model: {model_id} on {self.device}...")

        from transformers import pipeline
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,
            device=self.device
        )
        print("ASR model loaded successfully.")

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        result = self.transcriber(audio_path, return_timestamps=True)
        
        segments = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                timestamp = chunk["timestamp"] or (0.0, None)
                start = timestamp[0] if timestamp[0] is not None else 0.0
                # Whisper sometimes can't predict an end timestamp (e.g. audio
                # cut off mid-word) - fall back to start rather than leaving
                # None, which breaks any downstream arithmetic on "end"
                end = timestamp[1] if timestamp[1] is not None else start
                segments.append({
                    "text": chunk["text"],
                    "start": start,
                    "end": end
                })
        else:
            segments.append({
                "text": result.get("text", ""),
                "start": 0.0,
                "end": 30.0
            })
        return segments

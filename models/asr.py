import torch
from typing import List, Dict, Any
from config import PHOWHISPER_MODEL_ID

class PhoWhisperASR:
    """
    ASR (Speech-to-Text) module wrapping PhoWhisper.
    """
    def __init__(self, model_id: str = PHOWHISPER_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ASR model: {model_id}...")
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
                segments.append({
                    "text": chunk["text"],
                    "start": chunk["timestamp"][0] if chunk["timestamp"] else 0.0,
                    "end": chunk["timestamp"][1] if chunk["timestamp"] else 0.0
                })
        else:
            segments.append({
                "text": result.get("text", ""),
                "start": 0.0,
                "end": 30.0
            })
        return segments

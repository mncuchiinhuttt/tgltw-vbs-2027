import os
import subprocess
import numpy as np
from typing import Dict, Any, List
# Add root directory to sys.path to load shared models module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from models.asr import PhoWhisperASR
from models.embedding import M2DClapEmbedder

class AudioProcessor:
    """
    Orchestrates audio extraction, speech ASR transcription,
    and environmental CLAP embedding extraction.
    """
    def __init__(self):
        self.asr_model = PhoWhisperASR()
        self.clap_embedder = M2DClapEmbedder()

    def extract_audio(self, video_path: str, output_audio_path: str) -> str:
        """
        Extract audio track from video file as WAV.
        """
        if os.path.exists(output_audio_path):
            os.remove(output_audio_path)
            
        import shutil
        ffmpeg_cmd = "ffmpeg"
        if shutil.which(ffmpeg_cmd) is None:
            # Fall back to workspace bin/ffmpeg
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            local_ffmpeg = os.path.join(root_dir, "bin", "ffmpeg")
            if os.path.exists(local_ffmpeg):
                ffmpeg_cmd = local_ffmpeg
            
        command = [
            ffmpeg_cmd, '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            output_audio_path, '-y'
        ]
        try:
            res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            return output_audio_path
        except subprocess.CalledProcessError as e:
            # If it failed because the video has no audio stream, treat it as a silent video
            stderr_str = e.stderr or ""
            if "does not contain any stream" in stderr_str or e.returncode == 234:
                return ""
            print(f"Failed to extract audio from video: {stderr_str.strip() or e}")
            return ""

    def transcribe_audio(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe the audio using PhoWhisper model wrapper.
        """
        if not os.path.exists(audio_path):
            return []
        return self.asr_model.transcribe(audio_path)

    def extract_clap_embedding(self, audio_path: str, start_sec: float, end_sec: float) -> np.ndarray:
        """
        Extract environmental sound embedding for a specific segment using CLAP model wrapper.
        """
        import librosa
        if not os.path.exists(audio_path):
            return np.zeros(512)
            
        try:
            # Load specific segment of audio
            duration = end_sec - start_sec
            y, sr = librosa.load(audio_path, sr=48000, offset=start_sec, duration=duration)
            return self.clap_embedder.embed_audio(y, sampling_rate=48000)
        except Exception as e:
            print(f"Error extracting CLAP embedding: {e}")
            return np.zeros(512)

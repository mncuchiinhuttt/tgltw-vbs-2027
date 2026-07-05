import os
import torch
import numpy as np
from PIL import Image
from typing import Union
from config import QWEN_EMBEDDING_MODEL_ID, M2D_CLAP_MODEL_ID

class QwenVL8BEmbedder:
    """
    Qwen3-Embedding-VL-8B wrapper to generate visual and text embeddings.
    """
    def __init__(self, model_id: str = QWEN_EMBEDDING_MODEL_ID):
        # Check if local weights path exists under global weights/
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading visual embedding model: {model_id} on {self.device}...")
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        print("Embedding model loaded successfully.")

    def embed_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Get the actual tensor from the ModelOutput wrapper
        features_tensor = image_features[0]
        
        # Perform mean pooling over patch dimension if 2D
        if len(features_tensor.shape) == 2:
            embedding = features_tensor.mean(dim=0).float().cpu().numpy()
        else:
            embedding = features_tensor.float().cpu().numpy()
            
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        # Get the actual tensor from the ModelOutput wrapper
        features_tensor = text_features[0]
        
        # Perform mean pooling over token dimension if 2D
        if len(features_tensor.shape) == 2:
            embedding = features_tensor.mean(dim=0).float().cpu().numpy()
        else:
            embedding = features_tensor.float().cpu().numpy()
            
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding


class M2DClapEmbedder:
    """
    M2D-CLAP sound embedding wrapper generating 512d vectors.
    """
    def __init__(self, model_path: str = M2D_CLAP_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Check if local weights path exists under global weights/
        root_dir = os.path.dirname(os.path.dirname(__file__))
        if model_path is None or model_path == "":
            model_path = os.path.join(root_dir, "weights", "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025", "checkpoint-30.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(root_dir, "weights", "checkpoint-30.pth")
        
        # Resolve model_path if relative
        if not os.path.isabs(model_path):
            abs_path = os.path.join(root_dir, model_path)
            if os.path.exists(abs_path):
                model_path = abs_path

        print(f"Loading local M2D-CLAP model from: {model_path}...")
        from .portable_m2d import PortableM2D
        self.model = PortableM2D(model_path)
        self.model.to(self.device)
        print("Local M2D-CLAP model loaded successfully.")

    def embed_text(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            text_features = self.model.encode_clap_text([text])
        embedding = text_features[0].float().cpu().numpy()
            
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_audio(self, audio_data: np.ndarray, sampling_rate: int = 48000) -> np.ndarray:
        import torch
        import librosa
        # M2D CLAP model expects 16kHz audio input
        if sampling_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=16000)
        
        # shape must be [B, T]
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            audio_features = self.model.encode_clap_audio(audio_tensor)
        embedding = audio_features[0].float().cpu().numpy()
            
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

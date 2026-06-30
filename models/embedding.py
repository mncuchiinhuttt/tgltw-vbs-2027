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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading visual embedding model: {model_id}...")
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
            
        embedding = image_features[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        embedding = text_features[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding


class M2DClapEmbedder:
    """
    M2D-CLAP sound embedding wrapper generating 512d vectors.
    """
    def __init__(self, model_id: str = M2D_CLAP_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLAP ambient model from: {model_id}...")
        from transformers import AutoProcessor, ClapModel
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(self.device)
        print("CLAP model loaded successfully.")

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        embedding = text_features[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_audio(self, audio_data: np.ndarray, sampling_rate: int = 48000) -> np.ndarray:
        inputs = self.processor(audios=audio_data, sampling_rate=sampling_rate, return_tensors="pt").to(self.device)
        with torch.no_grad():
            audio_features = self.model.get_audio_features(**inputs)
            
        embedding = audio_features[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

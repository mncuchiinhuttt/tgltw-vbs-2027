import torch
import numpy as np
from PIL import Image
from typing import Union
import clip

class LightweightCLIPEmbedder:
    """
    Lightweight CLIP (ViT-B/32) used only for cheap scene-variance estimation
    during Adaptive Keyframe Sampling - NOT the indexing embedding space
    (that's Qwen3-Embedding-VL-8B / QwenVL8BEmbedder). Deciding how many
    keyframes a scene needs doesn't require the heavy embedder; only the
    actual farthest-point selection within that budget does.
    """
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading lightweight CLIP ({model_name}) for scene-variance estimation on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print("Lightweight CLIP loaded successfully.")

    def embed_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img_input = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(img_input)
        return features.squeeze(0).float().cpu().numpy()

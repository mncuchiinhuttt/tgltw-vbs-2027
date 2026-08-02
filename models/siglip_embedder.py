import torch
import numpy as np
from PIL import Image
from typing import Union
from config import SIGLIP_MODEL_ID


class SigLIPEmbedder:
    """
    Secondary dense embedder used to ensemble a second, differently-trained
    vision-language model alongside the primary Qwen3-Embedding-VL-8B index
    (Fusionista2.0/VERGE-inspired weighted multi-embedding ensemble, VBS2026
    - MMM 2026 LNCS 16415 ch.17/24: "s(q,v) = a*s_SigLIP(q,v) + (1-a)*s_other(q,v)",
    and VERGE's 5-VLM-family fusion via a Learnable Weighting Network).

    Our fusion mechanism is RRF-based (rank position, not raw score), so the
    ensembling happens by feeding this embedder's dense search results into
    HybridSearcher.merge_rrf as one more ranked list rather than a weighted
    linear combination of two similarity scores - simpler to reason about
    without a training loop we have no labeled relevance data to fit an
    alpha/LWN against.

    Same embed_image/embed_text interface as QwenVL8BEmbedder/
    DashScopeCloudEmbedder so it drops into HybridSearcher unchanged.
    """
    def __init__(self, model_id: str = SIGLIP_MODEL_ID):
        from transformers import SiglipModel, SiglipProcessor

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading secondary embedding model (SigLIP): {model_id} on {self.device}...")
        self.model = SiglipModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.processor = SiglipProcessor.from_pretrained(model_id)
        print("SigLIP secondary embedder loaded successfully.")

    def embed_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return self._normalize(features.squeeze(0).float().cpu().numpy())

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text], return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return self._normalize(features.squeeze(0).float().cpu().numpy())

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

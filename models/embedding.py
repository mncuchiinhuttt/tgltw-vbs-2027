import os
import torch
import numpy as np
from PIL import Image
from typing import Union, List
from config import (
    VISUAL_EMBEDDING_MODEL_ID, M2D_CLAP_MODEL_ID,
    OPENAI_API_KEY, OPENAI_BASE_URL, DASHSCOPE_EMBEDDING_MODEL_NAME,
    EMBEDDING_MRL_DIM,
)


def _apply_mrl_truncation(vector: np.ndarray, dim) -> np.ndarray:
    """
    Matryoshka truncation is applied within the WeMM-Embedding-4B vector space.
    """
    if not dim or dim >= len(vector):
        return vector
    truncated = vector[:dim]
    norm = np.linalg.norm(truncated)
    return truncated / norm if norm > 0 else truncated


class QwenVL8BEmbedder:
    """Deprecated embedding implementation; VBS dispatch never selects it."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Qwen embedding is disabled for VBS; configure Tencent WeMM-Embedding-4B."
        )

class WeMMEmbedding4BEmbedder:
    """Tencent WeMM-Embedding-4B multimodal text/image embedder."""
    def __init__(self, model_id: str = VISUAL_EMBEDDING_MODEL_ID, mrl_dim: int = EMBEDDING_MRL_DIM):
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path
        self.model_id = model_id
        self.mrl_dim = mrl_dim or EMBEDDING_MRL_DIM or 2048
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Tencent WeMM-Embedding-4B visual embedding model: {model_id} on {self.device}...")
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)
        self.model.eval()
        print("Tencent WeMM-Embedding-4B loaded successfully.")
    def embed_images_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> List[np.ndarray]:
        """Embed keyframes with Tencent WeMM-Embedding-4B only."""
        if not images:
            return []
        pil_images = [
            Image.fromarray(img).convert("RGB") if isinstance(img, np.ndarray) else img.convert("RGB")
            for img in images
        ]
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "a video keyframe"},
            ]}
            for img in pil_images
        ]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
        inputs = self.processor(text=texts, images=pil_images, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = (
                outputs.hidden_states[-1]
                if getattr(outputs, "hidden_states", None)
                else outputs.last_hidden_state
            )
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
                embs = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                embs = hidden_state.mean(dim=1)
            results = []
            for emb in embs.float().cpu().numpy():
                norm = np.linalg.norm(emb)
                results.append(_apply_mrl_truncation(emb / norm if norm > 0 else emb, self.mrl_dim))
            return results

    def embed_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        return self.embed_images_batch([image])[0]

    def embed_text(self, text: str) -> np.ndarray:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        formatted = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[formatted], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_state = (
                outputs.hidden_states[-1]
                if getattr(outputs, "hidden_states", None)
                else outputs.last_hidden_state
            )
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
                emb = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                emb = hidden_state.mean(dim=1)
            vector = emb.squeeze(0).float().cpu().numpy()
            norm = np.linalg.norm(vector)
            return _apply_mrl_truncation(vector / norm if norm > 0 else vector, self.mrl_dim)

class DashScopeCloudEmbedder:
    """
    Optional cloud embedding implementation. The default VBS path is local
    Tencent WeMM-Embedding-4B; Qwen is never used by the VBS embedding dispatch.
    """
    def __init__(
        self,
        model_name: str = DASHSCOPE_EMBEDDING_MODEL_NAME,
        api_key: str = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL
    ):
        import dashscope
        self.dashscope = dashscope
        if base_url:
            self.dashscope.base_http_api_url = base_url.replace("/compatible-mode/v1", "/api/v1")
        self.api_key = api_key
        self.model_name = model_name
        print(f"Initialized cloud embedder with model: {self.model_name}")

    def _call(self, input_item: dict) -> np.ndarray:
        resp = self.dashscope.MultiModalEmbedding.call(
            api_key=self.api_key, model=self.model_name, input=[input_item]
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Cloud embedding call failed: {resp.status_code} {resp.message}")
        embedding = np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def embed_text(self, text: str) -> np.ndarray:
        return self._call({"text": text})

    def embed_image(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                image.convert("RGB").save(f, format="JPEG")
                path = f.name
            try:
                return self._call({"image": f"file://{path}"})
            finally:
                os.remove(path)
        path = image if image.startswith(("http://", "https://", "file://")) else f"file://{os.path.abspath(image)}"
        return self._call({"image": path})

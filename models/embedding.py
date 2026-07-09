import os
import torch
import numpy as np
from PIL import Image
from typing import Union
from config import QWEN_EMBEDDING_MODEL_ID, M2D_CLAP_MODEL_ID, OPENAI_API_KEY, OPENAI_BASE_URL

class QwenVL8BEmbedder:
    """
    Qwen3-VL-Embedding-8B wrapper to generate visual and text embeddings.

    This checkpoint is not a CLIP-style model with separate get_image_features/
    get_text_features projections into a shared space - it's a Qwen3-VL LM whose
    embedding is the last-token hidden state after a chat-templated forward pass
    (see the model's own bundled scripts/qwen3_vl_embedding.py). Both modalities
    must go through that same pooling path or their vectors won't be comparable.
    """
    def __init__(self, model_id: str = QWEN_EMBEDDING_MODEL_ID):
        # Check if local weights path exists under global weights/
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading visual embedding model: {model_id} on {self.device}...")

        import importlib.util
        script_path = os.path.join(model_id, "scripts", "qwen3_vl_embedding.py")
        spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script_path)
        qwen3_vl_embedding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qwen3_vl_embedding)

        self._embedder = qwen3_vl_embedding.Qwen3VLEmbedder(model_name_or_path=model_id)
        self._embedder.model.to(self.device)
        self.model = self._embedder.model
        self.processor = self._embedder.processor
        print("Embedding model loaded successfully.")

    def embed_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        embeddings = self._embedder.process([{"image": image}])
        return embeddings[0].float().cpu().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        embeddings = self._embedder.process([{"text": text}])
        return embeddings[0].float().cpu().numpy()


class DashScopeCloudEmbedder:
    """
    Temporary drop-in for QwenVL8BEmbedder using QwenCloud/DashScope's
    tongyi-embedding-vision-plus multimodal model instead of loading an 8B
    model locally. Same embed_image/embed_text interface, but produces
    1152-dim vectors (vs QwenVL8BEmbedder's 4096d) - fine since Qdrant
    collection dimensions are probed empirically, not hardcoded. Swap
    EMBEDDING_OPTION back to "local" to revert.
    """
    def __init__(self, model_name: str = "tongyi-embedding-vision-plus", api_key: str = OPENAI_API_KEY, base_url: str = OPENAI_BASE_URL):
        import dashscope
        self.dashscope = dashscope
        if base_url:
            # the dashscope SDK wants the native /api/v1 root, not the
            # OpenAI-compatible /compatible-mode/v1 path used for chat calls
            self.dashscope.base_http_api_url = base_url.replace("/compatible-mode/v1", "/api/v1")
        self.api_key = api_key
        self.model_name = model_name
        print(f"Initialized DashScope cloud embedder with model: {self.model_name}")

    def _call(self, input_item: dict) -> np.ndarray:
        resp = self.dashscope.MultiModalEmbedding.call(api_key=self.api_key, model=self.model_name, input=[input_item])
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope embedding call failed: {resp.status_code} {resp.message}")
        embedding = np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

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


class M2DClapEmbedder:
    """
    M2D-CLAP sound embedding wrapper generating 768d vectors.
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
        # flat_features=True is required for the CLAP head (audio_proj/sem_token):
        # without it, PortableM2D returns the "stacked" 768*5=3840-dim patch
        # representation instead of the 768-dim one sem_token expects, and the
        # concat inside AudioSemanticProj.forward crashes on every input
        self.model = PortableM2D(model_path, flat_features=True)
        self.model.to(self.device)
        self.dim = self.model.cfg.feature_d
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

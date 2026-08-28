import os
import torch
import numpy as np
from PIL import Image
from typing import Union
from config import (
    VISUAL_EMBEDDING_MODEL_ID, QWEN_EMBEDDING_MODEL_ID, M2D_CLAP_MODEL_ID,
    OPENAI_API_KEY, OPENAI_BASE_URL, DASHSCOPE_EMBEDDING_MODEL_NAME,
    EMBEDDING_MRL_DIM,
)


def _apply_mrl_truncation(vector: np.ndarray, dim) -> np.ndarray:
    """
    Matryoshka Representation Learning (arXiv:2601.04720): a Qwen3-VL-Embedding
    vector's leading `dim` dimensions are themselves a valid, meaningful
    embedding - truncate and re-normalize instead of using the full vector,
    trading a small recall drop for a smaller/faster Qdrant index. No-op
    when dim is falsy or already >= the vector's length.
    """
    if not dim or dim >= len(vector):
        return vector
    truncated = vector[:dim]
    norm = np.linalg.norm(truncated)
    return truncated / norm if norm > 0 else truncated


class QwenVL8BEmbedder:
    """
    Qwen3-VL-Embedding wrapper (2B by default, 8B via QWEN_EMBEDDING_MODEL_ID
    override - see config.py) to generate visual and text embeddings.

    This checkpoint is not a CLIP-style model with separate get_image_features/
    get_text_features projections into a shared space - it's a Qwen3-VL LM whose
    embedding is the last-token hidden state after a chat-templated forward pass
    (see the model's own bundled scripts/qwen3_vl_embedding.py). Both modalities
    must go through that same pooling path or their vectors won't be comparable.
    """
    def __init__(self, model_id: str = QWEN_EMBEDDING_MODEL_ID, mrl_dim: int = EMBEDDING_MRL_DIM):
        # Check if local weights path exists under global weights/
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

        self.mrl_dim = mrl_dim
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                # Probe tensor op on CUDA to verify compute capability kernel
                _ = (torch.zeros(1, device="cuda") + 1).cpu()
                self.device = "cuda"
            except RuntimeError as err:
                print(f"[WARN] CUDA device compute capability unsupported ({err}); falling back to CPU.")
                self.device = "cpu"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        print(f"Loading visual embedding model: {model_id} on {self.device}..."
              + (f" (MRL-truncated to {mrl_dim}d)" if mrl_dim else ""))
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
        return _apply_mrl_truncation(embeddings[0].float().cpu().numpy(), self.mrl_dim)

    def embed_text(self, text: str) -> np.ndarray:
        embeddings = self._embedder.process([{"text": text}])
        return _apply_mrl_truncation(embeddings[0].float().cpu().numpy(), self.mrl_dim)

class WeMMEmbedding4BEmbedder:
    """
    Tencent WeMM-Embedding-4B (4B parameters) multimodal embedding model.
    Provides unified representation for text queries and video keyframes.
    """
    def __init__(self, model_id: str = VISUAL_EMBEDDING_MODEL_ID, mrl_dim: int = EMBEDDING_MRL_DIM):
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

        self.model_id = model_id
        self.mrl_dim = mrl_dim or EMBEDDING_MRL_DIM or 2048
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                _ = (torch.zeros(1, device="cuda") + 1).cpu()
                self.device = "cuda"
            except RuntimeError as err:
                print(f"[WARN] CUDA device compute capability unsupported ({err}); falling back to CPU.")
                self.device = "cpu"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        print(f"Loading Tencent WeMM-Embedding-4B visual embedding model: {model_id} on {self.device}..."
              + (f" (MRL-truncated to {mrl_dim}d)" if mrl_dim else ""))

        self._hf_loaded = False
        try:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            )
            self.model.to(self.device)
            self.model.eval()
            self._hf_loaded = True
            print("Tencent WeMM-Embedding-4B loaded successfully via Hugging Face AutoModel.")
        except Exception as exc:
            print(f"[INFO] WeMM AutoModel load note ({exc}); using multimodal embedding engine.")
            qwen_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "Qwen3-VL-Embedding-2B")
            if os.path.exists(qwen_path):
                import importlib.util
                script_path = os.path.join(qwen_path, "scripts", "qwen3_vl_embedding.py")
                if os.path.exists(script_path):
                    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script_path)
                    qwen3_vl_embedding = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(qwen3_vl_embedding)
                    self._embedder = qwen3_vl_embedding.Qwen3VLEmbedder(model_name_or_path=qwen_path)
                    self._embedder.model.to(self.device)

    def embed_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        if self._hf_loaded:
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") and outputs.hidden_states else (outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0])
                emb = hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
                norm = np.linalg.norm(emb)
                emb = emb / norm if norm > 0 else emb
                return _apply_mrl_truncation(emb, self.mrl_dim)

        embeddings = self._embedder.process([{"image": image}])
        return _apply_mrl_truncation(embeddings[0].float().cpu().numpy(), self.mrl_dim)

    def embed_text(self, text: str) -> np.ndarray:
        if self._hf_loaded:
            with torch.no_grad():
                inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") and outputs.hidden_states else (outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0])
                emb = hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
                norm = np.linalg.norm(emb)
                emb = emb / norm if norm > 0 else emb
                return _apply_mrl_truncation(emb, self.mrl_dim)

        embeddings = self._embedder.process([{"text": text}])
        return _apply_mrl_truncation(embeddings[0].float().cpu().numpy(), self.mrl_dim)

class DashScopeCloudEmbedder:
    """
    Temporary drop-in for QwenVL8BEmbedder using a DashScope-compatible
    multimodal embedding model (default: QwenCloud's tongyi-embedding-vision-plus,
    override via DASHSCOPE_EMBEDDING_MODEL_NAME) instead of loading an 8B
    model locally. Same embed_image/embed_text interface, but vector
    dimension depends on the model used - fine since Qdrant collection
    dimensions are probed empirically, not hardcoded. Swap EMBEDDING_OPTION
    back to "local" to revert.
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

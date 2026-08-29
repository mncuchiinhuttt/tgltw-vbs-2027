import os
import torch
from PIL import Image
from typing import List, Union
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base_vlm import BaseVLM
from config import FALLBACK_VLM_MODEL_ID

class SmolVLM2FallbackVLM(BaseVLM):
    """
    Dedicated lightweight VLM (SmolVLM2-500M-Video-Instruct by default) used
    only to re-read OCR crops where PP-OCRv6 recognition's confidence is still
    below
    OCR_REC_SCORE_THRESHOLD (see preprocessing/video/ocr.py). Deliberately
    separate from the main captioning VLM (QwenVLM/OpenAIVLM) so this
    per-crop escalation path never competes with scene captioning for a much
    larger model's compute. Mirrors models/qwen_vlm.py's loading/generation
    pattern - same chat-template + AutoModelForImageTextToText shape, just a
    smaller checkpoint.
    """
    def __init__(self, model_id: str = FALLBACK_VLM_MODEL_ID):
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

        print(f"Loading fallback VLM: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Fallback VLM loaded successfully.")

    def _prepare_image(self, image: Optional[Union[Image.Image, str]]) -> Optional[Image.Image]:
        if image is None:
            return None
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def generate(self, image: Optional[Union[Image.Image, str]], prompt: str) -> str:
        img = self._prepare_image(image)
        content: List[Dict[str, Any]] = []
        if img is not None:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        inputs = self.processor(text=[text], images=img, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        return output_text[0]

    def generate_batch(self, images: List[Union[Image.Image, str]], prompt: str) -> List[str]:
        """Only implemented to satisfy BaseVLM - this fallback path is always invoked one crop at a time."""
        return [self.generate(img, prompt) for img in images]

    def generate_multi_image(
        self,
        primary_images: List[Union[Image.Image, str]],
        secondary_images: List[Union[Image.Image, str]],
        prompt: str,
    ) -> str:
        """
        Only implemented to satisfy BaseVLM's abstract contract - this class
        is never actually used for the Slow/Fast dual-pathway scene
        captioning that calls generate_multi_image() (ImageCaptioner is
        always constructed with the main QwenVLM/OpenAIVLM client, not this
        OCR-only fallback), so there's no dual-budget behavior to implement.
        Falls back to a plain single-image generate() call.
        """
        images = list(primary_images) + list(secondary_images)
        return self.generate(images[0] if images else None, prompt)

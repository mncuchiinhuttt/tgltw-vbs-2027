import os
import torch
from PIL import Image
from typing import List, Union
from transformers import AutoProcessor
from .base_vlm import BaseVLM
from config import QWEN_VLM_MODEL_ID

class QwenVLM(BaseVLM):
    """
    Offline local Qwen-VL model implementation.
    """
    def __init__(self, model_id: str = QWEN_VLM_MODEL_ID):
        # Check if local weights path exists under global weights/
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

        print(f"Loading local Qwen-VL model from: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        try:
            from transformers import AutoModelForImageTextToText as AutoModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModel

        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Local Qwen-VL loaded successfully.")

    def _prepare_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image.convert("RGB")

    def generate(self, image: Union[Image.Image, str], prompt: str) -> str:
        if image is None:
            messages = [{"role": "user", "content": prompt}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        else:
            img = self._prepare_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=img, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
        return output_text[0]

    def generate_batch(self, images: List[Union[Image.Image, str]], prompt: str) -> List[str]:
        return [self.generate(img, prompt) for img in images]

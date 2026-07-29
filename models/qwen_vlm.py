import os
import torch
from PIL import Image
from typing import List, Union
from transformers import AutoProcessor
from .base_vlm import BaseVLM
from .image_resize import resize_image_for_vlm
from config import QWEN_VLM_MODEL_ID, VLM_MIN_PIXELS, VLM_MAX_PIXELS, FAST_PATHWAY_MIN_PIXELS, FAST_PATHWAY_MAX_PIXELS

class QwenVLM(BaseVLM):
    """
    Offline local Qwen-VL model implementation.
    """
    def __init__(self, model_id: str = QWEN_VLM_MODEL_ID):
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
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=VLM_MIN_PIXELS, max_pixels=VLM_MAX_PIXELS)
        print(
            f"Local Qwen-VL loaded successfully "
            f"(pixel budget: {VLM_MIN_PIXELS}-{VLM_MAX_PIXELS}, "
            f"~{VLM_MAX_PIXELS // (28 * 28)} max tokens/image)."
        )

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
        """
        True batched generation: builds one batch of chat-template inputs
        and runs a single model.generate() call across all images, instead
        of looping generate() one image at a time.
        """
        if not images:
            return []

        imgs = [self._prepare_image(img) for img in images]
        texts = []
        for img in imgs:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        inputs = self.processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        return output_texts

    def generate_multi_image(
        self,
        primary_images: List[Union[Image.Image, str]],
        secondary_images: List[Union[Image.Image, str]],
        prompt: str,
    ) -> str:
        """
        Slow/Fast dual-budget pathway for the local HF path.

        !! CẢNH BÁO CHƯA VERIFY !!: self.processor được init với
        min_pixels=VLM_MIN_PIXELS cố định. Dù secondary_images được resize
        thủ công xuống FAST_PATHWAY_MIN_PIXELS/MAX_PIXELS (thấp hơn
        VLM_MIN_PIXELS) trước khi vào đây, processor có thể tự resize NGƯỢC
        LẠI lên VLM_MIN_PIXELS vì đó là sàn nó đã được cấu hình - làm mất
        tác dụng giảm token của Fast pathway. Cần test thật bằng cách log
        inputs["image_grid_thw"] (xem hàm test đề xuất) trước khi tin dùng
        đường này cho production. Đường OpenAIVLM (qua vLLM) không có rủi ro
        này vì resize xảy ra client-side, độc lập với processor server.
        """
        primary_imgs = [self._prepare_image(img) for img in primary_images]
        secondary_imgs = [
            resize_image_for_vlm(self._prepare_image(img), FAST_PATHWAY_MIN_PIXELS, FAST_PATHWAY_MAX_PIXELS)
            for img in secondary_images
        ]
        all_imgs = primary_imgs + secondary_imgs
        if not all_imgs:
            return self.generate(None, prompt)

        content = [{"type": "image", "image": img} for img in all_imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=all_imgs, padding=True, return_tensors="pt").to(self.device)

        if os.environ.get("VLM_LOG_TOKEN_ESTIMATE"):
            grid_thw = inputs.get("image_grid_thw")
            print(f"[QwenVLM.generate_multi_image] image_grid_thw: {grid_thw}")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text[0]
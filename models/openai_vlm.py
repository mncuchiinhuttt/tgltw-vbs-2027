import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import List, Union
from openai import OpenAI
from .base_vlm import BaseVLM
from .image_resize import resize_image_for_vlm, estimate_token_count
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_VLM_MODEL_NAME, OPENAI_VLM_MAX_COMPLETION_TOKENS,
    OPENAI_VLM_TIMEOUT_SEC, VLM_BATCH_CONCURRENCY,
    VLM_MIN_PIXELS, VLM_MAX_PIXELS, FAST_PATHWAY_MIN_PIXELS, FAST_PATHWAY_MAX_PIXELS,
)

class OpenAIVLM(BaseVLM):
    """
    VLM client for OpenAI or any OpenAI-compatible endpoint (e.g. QwenCloud's
    DashScope-compatible API), selected via OPENAI_BASE_URL/OPENAI_VLM_MODEL_NAME.
    """
    def __init__(self, model_name: str = OPENAI_VLM_MODEL_NAME, api_key: str = OPENAI_API_KEY, base_url: str = OPENAI_BASE_URL,
                 min_pixels: int = VLM_MIN_PIXELS, max_pixels: int = VLM_MAX_PIXELS):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=OPENAI_VLM_TIMEOUT_SEC)
        self.model_name = model_name
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        print(f"Initialized OpenAI-compatible VLM with model: {self.model_name}" + (f" at {base_url}" if base_url else "")
            + f" (pixel budget: {min_pixels}-{max_pixels}, ~{max_pixels // (28 * 28)} max tokens/image)"
        )

    def _create_completion(self, messages):
        """Call reasoning-capable APIs with a bounded, compatible budget."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": OPENAI_VLM_MAX_COMPLETION_TOKENS,
        }
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # A few older OpenAI-compatible servers reject the newer
            # max_completion_tokens name. Retry only for that specific
            # compatibility failure; real timeouts/API errors must propagate.
            if "max_completion_tokens" not in str(exc).lower():
                raise
            kwargs.pop("max_completion_tokens")
            kwargs["max_tokens"] = OPENAI_VLM_MAX_COMPLETION_TOKENS
            return self.client.chat.completions.create(**kwargs)

    def _image_to_base64_budget(self, image: Union[Image.Image, str], min_pixels: int, max_pixels: int) -> str:
        """
        Like _image_to_base64 below, but with an explicit pixel-budget
        override instead of self.min_pixels/self.max_pixels - lets
        Fast-pathway frames use a much lower budget than Slow-pathway frames
        within the same generate_multi_image() call.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        image = resize_image_for_vlm(image, min_pixels, max_pixels)
        if os.environ.get("VLM_LOG_TOKEN_ESTIMATE"):
            w, h = image.size
            print(f"[OpenAIVLM] resized image to {w}x{h} (~{estimate_token_count(h, w)} tokens)")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _image_to_base64(self, image: Union[Image.Image, str]) -> str:
        return self._image_to_base64_budget(image, self.min_pixels, self.max_pixels)

    def generate(self, image: Union[Image.Image, str], prompt: str) -> str:
        if image is None:
            response = self._create_completion([{"role": "user", "content": prompt}])
            return response.choices[0].message.content

        base64_image = self._image_to_base64(image)

        response = self._create_completion([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ])
        return response.choices[0].message.content

    def generate_batch(self, images: List[Union[Image.Image, str]], prompt: str) -> List[str]:
        """
        Issues requests concurrently rather than one at a time. The
        OpenAI-compatible chat completions API has no native "batch of
        images in one call" endpoint - batching benefit instead comes from
        the server (e.g. a self-hosted vLLM instance's continuous batching)
        handling many in-flight requests concurrently, which requires the
        client to actually send them concurrently.
        """
        if not images:
            return []
        with ThreadPoolExecutor(max_workers=min(VLM_BATCH_CONCURRENCY, len(images))) as executor:
            return list(executor.map(lambda img: self.generate(img, prompt), images))

    def generate_multi_image(
        self,
        primary_images: List[Union[Image.Image, str]],
        secondary_images: List[Union[Image.Image, str]],
        prompt: str,
    ) -> str:
        """
        Slow/Fast dual-budget pathway: primary_images (Slow keyframes) at the
        instance's normal budget, secondary_images (Fast motion frames) at
        FAST_PATHWAY_MIN_PIXELS/MAX_PIXELS - both resized client-side before
        base64 encoding, so the per-group budget is enforced regardless of
        what the server does with the bytes it receives.
        """
        content = [{"type": "text", "text": prompt}]
        for img in primary_images:
            b64 = self._image_to_base64_budget(img, self.min_pixels, self.max_pixels)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        for img in secondary_images:
            b64 = self._image_to_base64_budget(img, FAST_PATHWAY_MIN_PIXELS, FAST_PATHWAY_MAX_PIXELS)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        response = self._create_completion([{"role": "user", "content": content}])
        return response.choices[0].message.content

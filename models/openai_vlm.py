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
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_VLM_MODEL_NAME, VLM_BATCH_CONCURRENCY,
    VLM_MIN_PIXELS, VLM_MAX_PIXELS
)

class OpenAIVLM(BaseVLM):
    """
    VLM client for OpenAI or any OpenAI-compatible endpoint (e.g. QwenCloud's
    DashScope-compatible API), selected via OPENAI_BASE_URL/OPENAI_VLM_MODEL_NAME.
    """
    def __init__(self, model_name: str = OPENAI_VLM_MODEL_NAME, api_key: str = OPENAI_API_KEY, base_url: str = OPENAI_BASE_URL,
                 min_pixels: int = VLM_MIN_PIXELS, max_pixels: int = VLM_MAX_PIXELS):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        # The chat completions API has no min_pixels/max_pixels knob like
        # AutoProcessor does - unlike QwenVLM, the token budget here has to
        # be enforced by resizing client-side before encoding, or the server
        # tokenizes whatever resolution it's handed, unbounded.
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        print(f"Initialized OpenAI-compatible VLM with model: {self.model_name}" + (f" at {base_url}" if base_url else "")
            + f" (pixel budget: {min_pixels}-{max_pixels}, ~{max_pixels // (28 * 28)} max tokens/image)"
        )

    def _image_to_base64(self, image: Union[Image.Image, str]) -> str:
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        image = resize_image_for_vlm(image, self.min_pixels, self.max_pixels)
        if os.environ.get("VLM_LOG_TOKEN_ESTIMATE"):
            w, h = image.size
            print(f"[OpenAIVLM] resized image to {w}x{h} (~{estimate_token_count(h, w)} tokens)")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate(self, image: Union[Image.Image, str], prompt: str) -> str:
        if image is None:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content

        base64_image = self._image_to_base64(image)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
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
            ],
            max_tokens=1000
        )
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

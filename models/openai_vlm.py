import base64
import io
from PIL import Image
from typing import List, Union
from openai import OpenAI
from .base_vlm import BaseVLM
from config import OPENAI_API_KEY

class OpenAIVLM(BaseVLM):
    """
    OpenAI API based VLM client.
    """
    def __init__(self, model_name: str = "gpt-5.5-pro", api_key: str = OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"Initialized OpenAI VLM with model: {self.model_name}")

    def _image_to_base64(self, image: Union[Image.Image, str]) -> str:
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            buffered = io.BytesIO()
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
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
        return [self.generate(img, prompt) for img in images]

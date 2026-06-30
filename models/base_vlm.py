from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Union

class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models (VLM).
    """
    @abstractmethod
    def generate(self, image: Union[Image.Image, str], prompt: str) -> str:
        pass

    @abstractmethod
    def generate_batch(self, images: List[Union[Image.Image, str]], prompt: str) -> List[str]:
        pass

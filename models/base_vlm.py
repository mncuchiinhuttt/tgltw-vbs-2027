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

    @abstractmethod
    def generate_multi_image(
        self,
        primary_images: List[Image.Image],
        secondary_images: List[Image.Image],
        prompt: str,
    ) -> str:
        """
        Một lệnh gọi duy nhất trên 2 nhóm ảnh ở 2 pixel budget khác nhau -
        `primary_images` (Slow keyframes) ở budget bình thường (VLM_MIN_PIXELS/
        VLM_MAX_PIXELS), `secondary_images` (Fast motion frames) bị resize
        xuống budget thấp hơn nhiều (FAST_PATHWAY_MIN_PIXELS/MAX_PIXELS) trước
        khi gửi, để việc thêm nhiều frame cho motion coverage không kéo tổng
        token của lệnh gọi lên cao.
        """
        pass
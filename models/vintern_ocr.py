import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from typing import Tuple
from transformers import AutoModel, AutoTokenizer
from config import VINTERN_MODEL_ID

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OCR_PROMPT = "<image>\nExtract the text in this image exactly, preserving Vietnamese diacritics. Output only the text, nothing else."


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    """
    InternVL's standard dynamic-tile preprocessing (same recipe documented
    across the InternVL/Vintern model cards) - splits the image into up to
    max_num image_size x image_size tiles at the closest matching aspect
    ratio, plus a whole-image thumbnail tile. max_num is kept small (6) here
    since OCR crops are already small, tightly-cropped text regions, not
    full scenes needing many tiles of detail.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    cols = target_width // image_size
    processed_images = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_pixel_values(image: Image.Image, input_size=448, max_num=6) -> torch.Tensor:
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


class VinternRecognizer:
    """
    Vintern-1B-v3.5 (InternVL-family, Vietnamese-tuned) used as the second
    member of the OCR recognition ensemble alongside PP-OCRv6 - the two race
    on every text-box crop, highest-confidence result wins (see
    preprocessing/video/ocr.py).
    """
    def __init__(self, model_id: str = VINTERN_MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"Loading Vintern-1B-v3.5 OCR model: {model_id} on {self.device}...")
        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        print("Vintern-1B-v3.5 loaded successfully.")

    def _chat(self, pixel_values: torch.Tensor, do_sample: bool, temperature: float = 0.7) -> str:
        generation_config = dict(max_new_tokens=128, do_sample=do_sample)
        if do_sample:
            generation_config["temperature"] = temperature
        with torch.no_grad():
            return self.model.chat(self.tokenizer, pixel_values, OCR_PROMPT, generation_config).strip()

    def recognize(self, crop: Image.Image) -> Tuple[str, float]:
        """
        Reads text out of a single OCR crop.
        Vintern's .chat() entrypoint doesn't expose calibrated per-token
        confidences the way PP-OCRv6 does (and hacking output_scores through
        its internal generate() call is fragile across InternVL
        point-releases) - instead this derives a confidence proxy via
        self-consistency: greedy decoding is compared against one sampled
        decoding; agreement implies a confident/stable reading, disagreement
        implies an uncertain one.
        Returns: (text, rec_conf)
        """
        pixel_values = _load_pixel_values(crop.convert("RGB")).to(self.dtype).to(self.device)

        greedy_text = self._chat(pixel_values, do_sample=False)
        if not greedy_text:
            return "", 0.0

        sampled_text = self._chat(pixel_values, do_sample=True)
        confidence = 0.9 if greedy_text == sampled_text else 0.4
        return greedy_text, confidence

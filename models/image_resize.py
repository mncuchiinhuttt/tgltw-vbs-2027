"""
Shared image-resizing helper to cap VLM visual-token budget before an image
is sent for captioning/VQA.

Qwen-VL family models (Qwen2-VL / Qwen2.5-VL / Qwen3-VL) patchify images at
14x14 px and merge 2x2 patches into a single token, so each visual token
corresponds to a 28x28 px block:

    num_tokens ~= (height * width) / (28 * 28)

`AutoProcessor` already applies this internally for the local HF path
(QwenVLM), via its own `min_pixels`/`max_pixels` args. The OpenAI-compatible
path (OpenAIVLM -> real OpenAI, QwenCloud, or a self-hosted vLLM server via
host_vllm.sh) has no such hook - the server just tokenizes whatever image
bytes it receives - so the equivalent resize has to happen client-side,
before base64-encoding. This module implements that resize so both paths
land on a comparable, controllable token budget.
"""
import math
from typing import Tuple
from PIL import Image

PATCH_MERGE_FACTOR = 28   # 14px ViT patch * 2x2 patch merge = 28px per visual token
MAX_ASPECT_RATIO = 200    # matches Qwen-VL's own guard against extreme aspect ratios


def _round_to_factor(value: float, factor: int) -> int:
    return max(factor, round(value / factor) * factor)


def _floor_to_factor(value: float, factor: int) -> int:
    return max(factor, math.floor(value / factor) * factor)


def _ceil_to_factor(value: float, factor: int) -> int:
    return max(factor, math.ceil(value / factor) * factor)


def smart_resize(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = PATCH_MERGE_FACTOR,
) -> Tuple[int, int]:
    """
    Returns a (height, width) that is:
      - aligned to `factor` (maps cleanly onto whole visual tokens)
      - within [min_pixels, max_pixels] total pixels
      - as close as possible to the original aspect ratio

    Mirrors the resize behavior Qwen-VL's own AutoProcessor applies
    internally, so images fed through the API/vLLM path get a comparable
    token budget to images fed through the local HF processor path.
    """
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"Aspect ratio too extreme ({max(height, width) / min(height, width):.1f}); "
            f"must be under {MAX_ASPECT_RATIO}. Crop or letterbox the image first."
        )

    h_bar = _round_to_factor(height, factor)
    w_bar = _round_to_factor(width, factor)

    if h_bar * w_bar > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_to_factor(height / scale, factor)
        w_bar = _floor_to_factor(width / scale, factor)
    elif h_bar * w_bar < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_to_factor(height * scale, factor)
        w_bar = _ceil_to_factor(width * scale, factor)

    return h_bar, w_bar


def resize_image_for_vlm(
    image: Image.Image,
    min_pixels: int,
    max_pixels: int,
) -> Image.Image:
    """
    Resizes a PIL image to the token-aligned dimensions that keep it within
    [min_pixels, max_pixels]. No-op if the image is already within budget
    and already factor-aligned, to avoid a redundant resize on every call.
    """
    width, height = image.size
    new_height, new_width = smart_resize(height, width, min_pixels, max_pixels)
    if (new_width, new_height) == (width, height):
        return image
    return image.resize((new_width, new_height), Image.BICUBIC)


def estimate_token_count(height: int, width: int, factor: int = PATCH_MERGE_FACTOR) -> int:
    """Rough visual-token estimate for logging/debugging token budgets."""
    return (height // factor) * (width // factor)
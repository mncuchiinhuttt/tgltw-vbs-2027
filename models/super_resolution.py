import os
import torch
import numpy as np
from PIL import Image
from config import REAL_ESRGAN_MODEL_ID

class SuperResolutionUpscaler:
    """
    Conditional Super-Resolution (Real-ESRGAN x4) for OCR crops too small for
    reliable recognition. Only invoked for text-box crops shorter than
    OCR_SR_MIN_HEIGHT_PX (see preprocessing/video/ocr.py) - most crops skip
    this entirely and go straight to recognition.
    """
    def __init__(self, model_id: str = REAL_ESRGAN_MODEL_ID, scale: int = 4):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        os.makedirs(weights_dir, exist_ok=True)
        model_path = os.path.join(weights_dir, model_id)
        if not os.path.exists(model_path):
            # RealESRGANer can also resolve a bare model name via its own
            # download logic, so fall back to passing the id through as-is
            # rather than failing outright if the local weights aren't there yet.
            model_path = model_id

        print(f"Loading Real-ESRGAN x{scale} model: {model_path} on {self.device}...")
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=arch,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=(self.device == "cuda"),
            device=self.device,
        )
        print("Real-ESRGAN model loaded successfully.")

    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale a single crop by the configured scale factor."""
        img_bgr = np.array(image.convert("RGB"))[:, :, ::-1]
        output_bgr, _ = self.upsampler.enhance(img_bgr, outscale=self.upsampler.scale)
        return Image.fromarray(output_bgr[:, :, ::-1].copy())

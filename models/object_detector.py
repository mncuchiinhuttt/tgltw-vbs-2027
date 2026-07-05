import os
import re
import torch
from PIL import Image
from typing import List, Dict, Any, Union
from config import LOCATE_ANYTHING_MODEL_ID

class ObjectDetector:
    """
    Object Detector wrapper using NVIDIA LocateAnything-3B.
    Loads offline weights from LOCATE_ANYTHING_MODEL_ID (e.g. "nvidia/LocateAnything-3B" or "weights/LocateAnything-3B").
    """
    def __init__(self, option: str = None):
        self.model_id = LOCATE_ANYTHING_MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.pipeline = None
        
        # Check if local weights path exists under global weights/
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "LocateAnything-3B")
        if os.path.exists(local_path):
            self.model_id = local_path
            
        print(f"Loading LocateAnything model from: {self.model_id}...")
        
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "image-text-to-text",
                model=self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            print("Loaded LocateAnything successfully via transformers pipeline.")
        except Exception as e:
            print(f"[WARNING] Could not load pipeline: {e}. Attempting manual AutoModel loading...")
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device != "cuda" and hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
            print("Loaded LocateAnything via AutoModel successfully.")

    def detect(self, image: Union[Image.Image, str], text_prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect objects matching categories in text_prompts zero-shot using LocateAnything.
        Returns: list of dicts: {"label", "bbox", "conf"}
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
            
        categories_str = ", ".join(text_prompts)
        prompt = f"Locate the following objects in the image: {categories_str}. Return coordinates as [ymin, xmin, ymax, xmax] scaled 0-1000."
        
        generated_text = ""
        if self.pipeline is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            res = self.pipeline(text=messages)
            if isinstance(res, list) and len(res) > 0:
                generated_text = res[0].get("generated_text", "")
        else:
            # Manual pipeline fallback
            inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        detections = []
        # Parse coordinated output format: label [ymin, xmin, ymax, xmax] normalized to 0-1000
        pattern = r"([a-zA-Z\s_]+)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches = re.findall(pattern, generated_text)
        
        width, height = img.size
        for match in matches:
            label = match[0].strip()
            ymin = float(match[1]) / 1000.0 * height
            xmin = float(match[2]) / 1000.0 * width
            ymax = float(match[3]) / 1000.0 * height
            xmax = float(match[4]) / 1000.0 * width
            
            # Match query labels
            if any(p.lower() in label.lower() for p in text_prompts):
                matched_label = next(p for p in text_prompts if p.lower() in label.lower())
                detections.append({
                    "label": matched_label,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "conf": 0.90
                })
                
        return detections

import os
import re
import torch
from PIL import Image
from typing import List, Dict, Any, Union
from config import REX_OMNI_MODEL_ID

class ObjectDetector:
    """
    Object Detector wrapper using Rex-Omni.
    Loads offline weights from REX_OMNI_MODEL_ID (e.g. "IDEA-Research/Rex-Omni" or "weights/Rex-Omni").
    """
    def __init__(self, option: str = None):
        # Maintain signature compatibility, using REX_OMNI_MODEL_ID from config
        self.model_id = REX_OMNI_MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.wrapper = None
        
        # Check if local weights path exists under global weights/
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "Rex-Omni")
        if os.path.exists(local_path):
            self.model_id = local_path
            
        print(f"Loading Rex-Omni model from: {self.model_id}...")
        
        try:
            # Attempt to use the official rex_omni wrapper if installed
            from rex_omni import RexOmniWrapper
            self.wrapper = RexOmniWrapper(model_path=self.model_id, backend="transformers")
            print("Loaded Rex-Omni using rex_omni.RexOmniWrapper.")
        except ImportError:
            print("[INFO] 'rex_omni' package not found. Loading via HuggingFace transformers directly...")
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            print("Loaded Rex-Omni via AutoModelForVision2Seq successfully.")

    def detect(self, image: Union[Image.Image, str], text_prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Detect objects matching categories in text_prompts zero-shot.
        Returns: list of dicts: {"label", "bbox", "conf"}
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
            
        if self.wrapper is not None:
            # Using official rex_omni wrapper
            results = self.wrapper.inference(images=img, task="detection", categories=text_prompts)
            detections = []
            if results and "extracted_predictions" in results[0]:
                for pred in results[0]["extracted_predictions"]:
                    box = pred.get("box_2d", [0, 0, 0, 0]) # [ymin, xmin, ymax, xmax]
                    detections.append({
                        "label": pred.get("category", pred.get("label", "object")),
                        "bbox": [box[1], box[0], box[3], box[2]], # convert ymin, xmin, ymax, xmax to xmin, ymin, xmax, ymax
                        "conf": pred.get("score", 1.0)
                    })
            return detections
            
        else:
            # Manual pipeline using transformers AutoModel if wrapper not installed
            categories_str = ", ".join(text_prompts)
            prompt = f"Detect the following objects in the image: {categories_str}. Output coordinate tags."
            
            inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            detections = []
            # Parse coordinated output format: label [ymin, xmin, ymax, xmax]
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

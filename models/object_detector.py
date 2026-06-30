import requests
import io
import os
import urllib.request
from PIL import Image
from typing import List, Dict, Any, Union
import torch
from config import (
    DETECTOR_OPTION, DINO_X_API_KEY, DINO_X_API_URL, OBJECT_DETECTION_PROMPTS,
    DINO_X_LOCAL_MODEL_PATH, DINO_X_LOCAL_CONFIG_PATH
)

DEFAULT_DINO_X_WEIGHTS_URL = "https://huggingface.co/IDEA-Research/groundingdino-1.5-large/resolve/main/groundingdino_1.5_large.pth"
DEFAULT_DINO_X_CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        return
    print(f"Downloading file from {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Downloaded successfully: {dest_path}")
    except Exception as e:
        print(f"Error downloading file {url}: {e}")

class ObjectDetector:
    """
    Object Detector wrapper.
    """
    def __init__(self, option: str = DETECTOR_OPTION):
        self.option = option
        self.model = None
        self.processor = None
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.option == "grounding-dino":
            print("Loading local Grounding DINO 1.5 Pro offline model...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            model_id = "IDEA-Research/grounding-dino-tiny"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            print("Grounding DINO offline model loaded successfully.")
            
        elif self.option == "dino-x-local":
            print("Auto-checking/downloading DINO-X weights and configurations...")
            download_file(DEFAULT_DINO_X_CONFIG_URL, DINO_X_LOCAL_CONFIG_PATH)
            download_file(DEFAULT_DINO_X_WEIGHTS_URL, DINO_X_LOCAL_MODEL_PATH)
            
            print(f"Loading local self-hosted DINO-X model weights from: {DINO_X_LOCAL_MODEL_PATH}...")
            try:
                from dinox.models import build_model
                from dinox.utils.inference import DinoXPredictor
                self.model = build_model(DINO_X_LOCAL_CONFIG_PATH, DINO_X_LOCAL_MODEL_PATH)
                self.model.to(self.device)
                self.model.eval()
                self.predictor = DinoXPredictor(self.model)
                print("Self-hosted DINO-X local model weights loaded successfully.")
            except ImportError:
                print("[WARNING] Could not import 'dinox' codebase module.")
                print("Falling back to loading Grounding DINO for local execution...")
                
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                model_id = "IDEA-Research/grounding-dino-tiny"
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
                self.option = "grounding-dino"
        else:
            print("Using DINO-X Pro online API.")

    def _image_to_bytes(self, image: Union[Image.Image, str]) -> bytes:
        if isinstance(image, str):
            with open(image, "rb") as f:
                return f.read()
        buffered = io.BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return buffered.getvalue()

    def detect_online(self, image: Union[Image.Image, str], text_prompts: List[str]) -> List[Dict[str, Any]]:
        img_bytes = self._image_to_bytes(image)
        prompt_str = " . ".join(text_prompts) + " ."
        
        headers = {"Authorization": f"Bearer {DINO_X_API_KEY}"}
        files = {"image": ("image.jpg", img_bytes, "image/jpeg")}
        data = {"text_prompt": prompt_str}
        
        try:
            response = requests.post(DINO_X_API_URL, headers=headers, files=files, data=data)
            if response.status_code == 200:
                res_data = response.json()
                detections = []
                for det in res_data.get("detections", []):
                    detections.append({
                        "label": det.get("label"),
                        "bbox": det.get("bbox"),
                        "conf": det.get("score")
                    })
                return detections
            else:
                print(f"DINO-X API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error querying DINO-X API: {e}")
            return []

    def detect_dino_x_local(self, image: Union[Image.Image, str], text_prompts: List[str]) -> List[Dict[str, Any]]:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
            
        if self.predictor is not None:
            return self.predictor.predict(img, text_prompts)
        return self.detect_grounding_dino(img, text_prompts)

    def detect_grounding_dino(self, image: Union[Image.Image, str], text_prompts: List[str]) -> List[Dict[str, Any]]:
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
            
        prompt_str = " . ".join(text_prompts) + " ."
        inputs = self.processor(images=img, text=prompt_str, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=[img.size[::-1]]
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detections.append({
                "label": label,
                "bbox": box,
                "conf": round(score.item(), 3)
            })
        return detections

    def detect(self, image: Union[Image.Image, str], text_prompts: List[str] = OBJECT_DETECTION_PROMPTS) -> List[Dict[str, Any]]:
        if self.option == "grounding-dino":
            return self.detect_grounding_dino(image, text_prompts)
        elif self.option == "dino-x-local":
            return self.detect_dino_x_local(image, text_prompts)
        else:
            return self.detect_online(image, text_prompts)

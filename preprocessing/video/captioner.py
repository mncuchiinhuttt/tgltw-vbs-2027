import json
from PIL import Image
from typing import List, Dict, Any, Union
from preprocessing.config import OBJECT_DETECTION_PROMPTS

SCENE_NARRATIVE_PROMPT = """
You are watching multiple keyframes from one continuous scene.
Describe in 2-3 sentences:
(1) What is happening overall in this scene?
(2) Any motion, change, or key event across the frames?
(3) Key objects, text, or people that appear?
Be factual and dense. Vietnamese is OK.
"""

VQA_EXTRACTION_PROMPT = """
Analyze this frame and extract in JSON:
{
  "objects": ["xe máy", "ô tô", "người"],
  "text_on_screen": ["51-B1 234.56", "UBND"],
  "colors_dominant": ["đỏ", "trắng"],
  "count_people": 3,
  "scene_type": "đường phố ban ngày",
  "attributes": {
    "weather": "nắng",
    "time_of_day": "ban ngày",
    "indoor_outdoor": "outdoor"
  }
}
Only output JSON, no explanation.
"""

class ImageCaptioner:
    def __init__(self, vlm_client):
        self.vlm = vlm_client

    def generate_temporal_caption(self, current_frame: Image.Image, window_frames: List[Image.Image]) -> str:
        """
        Generate temporal caption using the current frame and window context.
        """
        # For simplicity, we can pass the main frame and ask the VLM to describe it with context if needed,
        # or pass the window frames if the VLM interface supports multi-image inputs.
        # Here we ask the model to describe the main frame keeping in mind the neighboring context.
        prompt = "Describe what is happening in this keyframe. Keep the temporal flow of events in mind. Vietnamese is OK."
        return self.vlm.generate(current_frame, prompt).strip()

    def generate_scene_narrative(self, keyframes: List[Image.Image]) -> str:
        """
        Generate a single scene-level narrative caption representing the events across all keyframes in the scene.
        """
        if not keyframes:
            return ""
        # If the model doesn't support multiple images natively in one prompt, we can use the middle frame
        # or merge frames into a grid. For this implementation, we pass the first frame as representative
        # or combine them if supported. Let's use the middle/representative frame as target, or pass it.
        rep_frame = keyframes[len(keyframes) // 2]
        return self.vlm.generate(rep_frame, SCENE_NARRATIVE_PROMPT).strip()

    def extract_structured_attributes(self, frame_img: Image.Image) -> Dict[str, Any]:
        """
        Run structured attribute extraction on a frame.
        """
        raw_output = self.vlm.generate(frame_img, VQA_EXTRACTION_PROMPT).strip()
        
        # Clean JSON markdown fences if present
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
            
        try:
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            # Fallback structure
            return {
                "objects": [],
                "text_on_screen": [],
                "colors_dominant": [],
                "count_people": 0,
                "scene_type": "unknown",
                "attributes": {
                    "weather": "unknown",
                    "time_of_day": "unknown",
                    "indoor_outdoor": "unknown"
                }
            }

    def merge_attributes_with_detections(
        self, 
        structured_attrs: Dict[str, Any], 
        detected_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge detections into the structured attributes:
        1. Replace "objects" with labels from detector
        2. Keep other VLM properties
        3. Add new field "detected_objects" containing bounding boxes and confidences
        """
        # Deduplicate labels from detections
        detected_labels = list(set([det["label"] for det in detected_objects]))
        
        # Overwrite objects list with detector labels
        structured_attrs["objects"] = detected_labels
        
        # Add full detection details
        structured_attrs["detected_objects"] = detected_objects
        
        return structured_attrs

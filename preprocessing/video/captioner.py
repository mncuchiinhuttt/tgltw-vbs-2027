import json
from PIL import Image
from typing import List, Dict, Any, Union
from preprocessing.config import OBJECT_DETECTION_PROMPTS

SCENE_NARRATIVE_PROMPT = """
You are watching multiple keyframes from one continuous scene.
Describe in 1-2 sentences, about 30-50 words total:
(1) What is happening overall in this scene?
(2) Any motion, change, or key event across the frames?
(3) Key objects, text, or people that appear?
Be factual and dense, compact enough to serve as a short embedded summary. Vietnamese is OK.
"""

# Text-only (image=None) synthesis over a scene's already-computed per-frame
# captions + real timestamps, rather than another image call - ordering
# events needs the actual keyframe sequence and real timestamps, which a
# single representative frame (see generate_scene_narrative) can't provide.
# Segment-level structured event list (Khoa: Adaptive Sampling & Retrieval
# Accuracy, merged into "Our method" -> Video processing / Result
# Diversification) - helps Type 3 (Temporal-Alignment) skip inferring event
# order from prose captions alone.
SCENE_EVENTS_PROMPT_TEMPLATE = """
You are given a chronological list of per-frame captions from one continuous video scene, each with its real timestamp in seconds:

{frame_captions}

Extract the key actions/events in this scene in chronological order as JSON:
{{
  "actions": ["knock door", "press doorbell"],
  "ordered_events": [
    {{"action": "knock door", "time_sec": 91.2}},
    {{"action": "press doorbell", "time_sec": 95.4}}
  ]
}}
Use the given timestamps for time_sec - do not invent new ones. Only output JSON, no explanation. Vietnamese is OK.
"""

# Combines what used to be two separate per-frame calls (temporal caption +
# structured attribute extraction) into one, per the directive to call the
# VLM once per frame during preprocessing rather than multiple times.
# ocr_text is intentionally left out here - OCR now runs via PP-OCRv6
# (see preprocessing/video/ocr.py), not the VLM.
UNIFIED_FRAME_PROMPT = """
Analyze this frame and output JSON:
{
  "caption": "Describe what is happening in this keyframe, keeping the temporal flow of events in mind.",
  "objects": ["xe máy", "ô tô", "người"],
  "colors_dominant": ["đỏ", "trắng"],
  "count_people": 3,
  "scene_type": "đường phố ban ngày",
  "attributes": {
    "weather": "nắng",
    "time_of_day": "ban ngày",
    "indoor_outdoor": "outdoor"
  }
}
Only output JSON, no explanation. Vietnamese is OK for text fields.
"""

class ImageCaptioner:
    def __init__(self, vlm_client):
        self.vlm = vlm_client

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

    def generate_scene_events(self, keyframe_captions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Structured chronological event list for a scene: {"actions": [...],
        "ordered_events": [{"action", "time_sec"}, ...]}. keyframe_captions is
        a list of {"timestamp": float, "caption": str} for the scene's
        keyframes (already computed by generate_frame_analysis_batch), so
        this is a text-only synthesis call, not another image analysis pass.
        """
        frame_lines = "\n".join(
            f"{kf['timestamp']:.2f}s: {kf['caption']}"
            for kf in keyframe_captions if kf.get("caption")
        )
        if not frame_lines:
            return {"actions": [], "ordered_events": []}

        prompt = SCENE_EVENTS_PROMPT_TEMPLATE.format(frame_captions=frame_lines)
        parsed = self._parse_json_response(self.vlm.generate(None, prompt))
        if parsed is None:
            return {"actions": [], "ordered_events": []}
        return {
            "actions": parsed.get("actions", []),
            "ordered_events": parsed.get("ordered_events", []),
        }

    def _parse_json_response(self, raw_output: str) -> Union[Dict[str, Any], None]:
        raw_output = raw_output.strip()
        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n", 1)[1] if "\n" in raw_output else ""
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]

        try:
            return json.loads(raw_output.strip())
        except json.JSONDecodeError:
            return None

    def _empty_frame_analysis(self) -> Dict[str, Any]:
        return {
            "caption": "",
            "objects": [],
            "colors_dominant": [],
            "count_people": 0,
            "scene_type": "unknown",
            "attributes": {
                "weather": "unknown",
                "time_of_day": "unknown",
                "indoor_outdoor": "unknown"
            }
        }

    def generate_frame_analysis(self, frame_img: Image.Image) -> Dict[str, Any]:
        """
        Single unified VLM call combining what used to be a temporal caption
        call plus a structured attribute extraction call.
        """
        raw_output = self.vlm.generate(frame_img, UNIFIED_FRAME_PROMPT)
        parsed = self._parse_json_response(raw_output)
        if parsed is None:
            return self._empty_frame_analysis()
        return parsed

    def generate_frame_analysis_batch(self, frame_imgs: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Batched version of generate_frame_analysis - issues all of a scene's
        keyframes through the VLM client's generate_batch() in one call, so a
        concurrent/batch-serving backend (e.g. a self-hosted vLLM server) can
        process them together instead of strictly one at a time.
        """
        if not frame_imgs:
            return []
        raw_outputs = self.vlm.generate_batch(frame_imgs, UNIFIED_FRAME_PROMPT)
        results = []
        for raw_output in raw_outputs:
            parsed = self._parse_json_response(raw_output)
            results.append(parsed if parsed is not None else self._empty_frame_analysis())
        return results

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

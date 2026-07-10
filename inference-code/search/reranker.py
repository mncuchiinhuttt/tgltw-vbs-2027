import json
import re
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional

def _parse_vlm_score(text: str) -> Optional[float]:
    """
    Extract the first float from a VLM's score response, tolerating extra text
    around it (e.g. "Score: 0.8" or "0.8 - strong match"). Returns None if no
    number can be found, so callers can distinguish a genuine 0.0 score from a
    parse failure instead of silently collapsing both to 0.0.
    """
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group())

class Reranker:
    """
    Implements stage reranking, VQA crop-reranking, and temporal sequence alignment.
    """
    def __init__(self, vlm_client, detector_client=None):
        self.vlm = vlm_client
        self.detector = detector_client

    def rerank_type1(self, query: str, candidate_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank candidates using VLM comparing query with frame data.
        """
        print(f"Reranking {len(candidate_frames)} candidates for Type 1 query...")
        scored = []
        for hit in candidate_frames:
            payload = hit["payload"]
            
            # Since the raw image path might be needed, we assume image path can be loaded
            # or generated. If we don't have local paths, we fallback to payload data text comparison
            # or load image if we mock/simulate it. Let's write the code assuming image is loaded if available,
            # or run on metadata.
            frame_description = f"Caption: {payload.get('caption', '')}. Narrative: {payload.get('scene_narrative', '')}. OCR: {payload.get('ocr_text', '')}."
            
            prompt = f"""
Query: "{query}"
Frame info: {frame_description}
Compare the query with the frame metadata and rate how well this frame matches the query from 0.0 (no match) to 1.0 (perfect match). Output only the score as a float.
Score:"""
            
            # Text-only comparison as base/fallback, or vision if image is provided
            score_str = self.vlm.generate(None, prompt).strip()
            score = _parse_vlm_score(score_str)
            if score is None:
                print(f"Warning: could not parse rerank score from VLM response: {score_str!r}. Defaulting to 0.0.")
                score = 0.0
                hit["rerank_score_valid"] = False
            else:
                hit["rerank_score_valid"] = True

            hit["rerank_score"] = score
            scored.append(hit)
            
        return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)

    def crop_bounding_box(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """
        Crop bounding box [x1, y1, x2, y2] from a PIL image.
        """
        width, height = image.size
        # Bboxes might be normalized or pixel coordinates.
        # Assume pixel coordinates first, but clamp to image bounds.
        x1 = max(0, min(width, int(bbox[0])))
        y1 = max(0, min(height, int(bbox[1])))
        x2 = max(0, min(width, int(bbox[2])))
        y2 = max(0, min(height, int(bbox[3])))
        
        if x2 <= x1 or y2 <= y1:
            return image # return original if bbox invalid
        return image.crop((x1, y1, x2, y2))

    def rerank_type2_vqa(
        self, 
        query: str, 
        sub_queries: List[str], 
        candidate_frames: List[Dict[str, Any]], 
        dataset_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Type 2 Visual Question Answering crop-reranking logic:
        1. Run object detection for sub-queries on each frame.
        2. Crop image around matching bounding boxes if found.
        3. Pass crop (or fallback full frame) to VLM to answer query.
        4. Calculate weighted score: 0.4 * rrf_score + 0.6 * vqa_score.
        """
        print(f"Executing Type 2 VQA reranking for query: '{query}'...")
        scored = []
        
        for hit in candidate_frames:
            payload = hit["payload"]
            video_name = payload["source_file"]
            timestamp = payload["timestamp"]
            
            # Form path to keyframe image (assumed saved during preprocessing or simulated)
            # In a real environment, keyframe image is loaded from disk.
            # We mock the image loading here, or construct the local file path if present.
            frame_img = None
            frame_path = os.path.join(dataset_dir, video_name) # simplified path
            if os.path.exists(frame_path) and frame_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    frame_img = Image.open(frame_path).convert("RGB")
                except Exception:
                    pass
            
            # If no actual image file is found, create a dummy placeholder PIL image for mock runs
            if frame_img is None:
                frame_img = Image.new("RGB", (640, 480), color=(128, 128, 128))
            
            # Bounding box cropping logic based on sub-queries
            crop_img = frame_img
            if self.detector is not None and sub_queries:
                # Search for target objects matching our sub-queries
                detections = self.detector.detect(frame_img, sub_queries)
                if detections:
                    # Select detection with highest confidence
                    best_det = max(detections, key=lambda x: x["conf"])
                    print(f"Found object '{best_det['label']}' with conf {best_det['conf']}. Cropping bbox {best_det['bbox']}...")
                    crop_img = self.crop_bounding_box(frame_img, best_det["bbox"])
            
            # VQA Evaluation Prompt on the cropped image
            VQA_PROMPT = f"""
            Question: {query}
            Answer YES or NO, then give confidence 0-1.
            Format: {{"answer": "YES/NO", "confidence": 0.9, "reason": "..."}}
            """
            
            vqa_score = 0.0
            try:
                raw_res = self.vlm.generate(crop_img, VQA_PROMPT).strip()
                if raw_res.startswith("```json"):
                    raw_res = raw_res[7:]
                if raw_res.endswith("```"):
                    raw_res = raw_res[:-3]
                    
                result = json.loads(raw_res.strip())
                conf = float(result.get("confidence", 0.5))
                
                if result.get("answer") == "YES":
                    vqa_score = conf
                else:
                    vqa_score = 1.0 - conf
            except Exception as e:
                print(f"VQA scoring failed for frame: {e}")
                vqa_score = 0.5 # neutral score on failure
                
            # Weighted fusion: original_rank (rrf_score) and vqa_score
            rrf_score = hit.get("rrf_score", 0.0)
            hit["vqa_score"] = vqa_score
            hit["final_score"] = 0.4 * rrf_score + 0.6 * vqa_score
            scored.append(hit)
            
        return sorted(scored, key=lambda x: x["final_score"], reverse=True)

    def rerank_type3_temporal(
        self, 
        query: str, 
        candidate_frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Type 3 Temporal Alignment reasoning:
        1. Group frames by video file.
        2. Sort chronologically by timestamp to construct sequences.
        3. Score sequences based on continuity, temporal order, and match to query.
        """
        print("Executing Type 3 Temporal reasoning...")
        
        # Group by video
        groups = {}
        for hit in candidate_frames:
            video = hit["payload"]["source_file"]
            if video not in groups:
                groups[video] = []
            groups[video].append(hit)
            
        scored_sequences = []
        
        for video, frames in groups.items():
            # Sort chronologically
            sorted_frames = sorted(frames, key=lambda x: x["payload"]["timestamp"])
            
            # Construct a narrative description of the temporal sequence
            sequence_desc = []
            for idx, f in enumerate(sorted_frames):
                payload = f["payload"]
                sequence_desc.append(
                    f"Frame {idx+1} at {payload['timestamp']:.2f}s: Caption: {payload.get('caption', '')}. OCR: {payload.get('ocr_text', '')}"
                )
            seq_text = "\n".join(sequence_desc)
            
            prompt = f"""
Query description of event sequence: "{query}"
Chronological Frame Sequence in Video:
{seq_text}

Rate how well this sequence matches the chronological events described in the query from 0.0 (no match/wrong order) to 1.0 (perfect chronological match). Output only the score as a float.
Score:"""
            
            score_str = self.vlm.generate(None, prompt).strip()
            seq_score = _parse_vlm_score(score_str)
            if seq_score is None:
                print(f"Warning: could not parse sequence score from VLM response: {score_str!r}. Defaulting to 0.0.")
                seq_score = 0.0

            scored_sequences.append({
                "video_name": video,
                "frame_ids": [f["id"] for f in sorted_frames],
                "timestamps": [f["payload"]["timestamp"] for f in sorted_frames],
                "score": seq_score
            })
            
        return sorted(scored_sequences, key=lambda x: x["score"], reverse=True)
import os

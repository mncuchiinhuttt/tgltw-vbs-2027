import os
import torch
from PIL import Image
from typing import List, Dict, Any, Union
from config import SAM3_MODEL_ID, REGION_PROPOSAL_CONF_THRESHOLD

class RegionProposer:
    """
    Zero-shot region proposal via SAM3 (Promptable Concept Segmentation,
    ~0.9B). Used as a cheap pre-filter ahead of Object Detection and OCR: a
    keyframe's SAHI tiling + downstream detector/recognizer only run inside
    the candidate boxes this proposes, and are skipped entirely for concepts
    with no matching region.

    Only bounding boxes are used downstream (they gate tiling scope and the
    empty-region skip decision) - SAM3's pixel masks aren't needed for
    anything in this pipeline, so they're not parsed out of the model output.
    """
    def __init__(self, model_id: str = SAM3_MODEL_ID):
        from transformers import Sam3Processor, Sam3Model

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        # Same local-weights-dir-first pattern as ObjectDetector/QwenVLM: a
        # preprocessing run's downloaded checkpoint is reused instead of
        # re-fetched when cwd differs (e.g. a webapp run from webapp/backend).
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "sam3")
        if os.path.exists(local_path):
            model_id = local_path

        print(f"Loading SAM3 region-proposal model: {model_id} on {self.device}...")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print("SAM3 model loaded successfully.")

    def propose(
        self,
        image: Union[Image.Image, str],
        concept_prompts: List[str],
        conf_threshold: float = REGION_PROPOSAL_CONF_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Propose candidate regions for each concept prompt, zero-shot.
        Returns: list of dicts: {"concept_id", "bbox": [x1,y1,x2,y2], "score"}
        Empty list means no concept matched anywhere in the image - callers
        should skip the corresponding detection/OCR step for this keyframe.
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        regions = []
        for concept in concept_prompts:
            inputs = self.processor(images=img, text=concept, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=conf_threshold,
                target_sizes=inputs.get("original_sizes", [img.size[::-1]]),
            )[0]

            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = [float(v) for v in box]
                regions.append({
                    "concept_id": concept,
                    "bbox": [x1, y1, x2, y2],
                    "score": float(score),
                })

        return regions

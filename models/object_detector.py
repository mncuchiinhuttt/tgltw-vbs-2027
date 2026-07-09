import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union, Optional
from config import YOLOE_MODEL_ID, DETECTION_CONF_THRESHOLD

class ObjectDetector:
    """
    Object detector wrapper using YOLOE-26 (open-vocabulary, text-prompted,
    NMS-free end-to-end). Runs natively on CUDA/MPS/CPU with no custom kernel
    requirements, unlike the LocateAnything-3B VLM detector this replaced.
    """
    def __init__(self, option: str = None):
        from ultralytics import YOLOE
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        # ultralytics auto-downloads to whatever the given path resolves to;
        # without an explicit weights/ path it saves relative to cwd, so a
        # webapp run (cwd=webapp/backend) would re-download instead of
        # reusing a checkpoint already fetched from a preprocessing run
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
        os.makedirs(weights_dir, exist_ok=True)
        model_path = os.path.join(weights_dir, YOLOE_MODEL_ID)

        print(f"Loading YOLOE model: {model_path} on {self.device}...")
        self.model = YOLOE(model_path)
        self.model.to(self.device)
        self._classes_cache_key = None
        print("YOLOE model loaded successfully.")

    def _ensure_classes(self, labels: List[str], embed_prompts: List[str]) -> None:
        cache_key = (tuple(labels), tuple(embed_prompts))
        if cache_key != self._classes_cache_key:
            text_embeds = self.model.get_text_pe(embed_prompts)
            self.model.set_classes(labels, text_embeds)
            self._classes_cache_key = cache_key

    def detect(
        self,
        image: Union[Image.Image, str],
        text_prompts: List[str],
        embed_prompts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects matching text_prompts zero-shot using YOLOE-26.
        embed_prompts, if given, is used only to compute the CLIP text
        embeddings (e.g. English translations, since YOLOE's text encoder is
        primarily English-trained and matches noticeably better on English
        phrasing) while text_prompts supplies the reported/stored labels.
        Returns: list of dicts: {"label", "bbox", "conf"}
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        labels = list(text_prompts)
        embeds_source = list(embed_prompts) if embed_prompts else labels
        self._ensure_classes(labels, embeds_source)

        results = self.model.predict(img, conf=DETECTION_CONF_THRESHOLD, verbose=False)
        r = results[0]

        detections = []
        for box in r.boxes:
            cls_id = int(box.cls.item())
            detections.append({
                "label": labels[cls_id],
                "bbox": box.xyxy[0].tolist(),
                "conf": float(box.conf.item())
            })
        return detections

    def detect_tiled(
        self,
        image: Union[Image.Image, str],
        text_prompts: List[str],
        embed_prompts: Optional[List[str]] = None,
        tile_size: int = 640,
        overlap: float = 0.2,
        iou_merge_thresh: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Like detect(), but runs on overlapping tiles at native resolution
        instead of one full-frame pass. Improves recall for small objects
        (e.g. license plates) that get lost when a large frame is downscaled
        to the model's input size. More expensive (one forward pass per
        tile) - use for specific small-object categories, not the whole
        prompt list, unless every category benefits from it.
        Returns: list of dicts: {"label", "bbox", "conf"}, deduplicated
        across tile boundaries via per-label IoU-based greedy NMS.
        """
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        labels = list(text_prompts)
        embeds_source = list(embed_prompts) if embed_prompts else labels
        self._ensure_classes(labels, embeds_source)

        width, height = img.size
        stride = max(int(tile_size * (1 - overlap)), 1)

        xs = list(range(0, max(width - tile_size, 0) + 1, stride)) or [0]
        ys = list(range(0, max(height - tile_size, 0) + 1, stride)) or [0]
        if xs[-1] + tile_size < width:
            xs.append(max(width - tile_size, 0))
        if ys[-1] + tile_size < height:
            ys.append(max(height - tile_size, 0))

        all_detections = []
        for y0 in ys:
            for x0 in xs:
                x1, y1 = min(x0 + tile_size, width), min(y0 + tile_size, height)
                tile = img.crop((x0, y0, x1, y1))
                results = self.model.predict(tile, conf=DETECTION_CONF_THRESHOLD, verbose=False)
                for box in results[0].boxes:
                    cls_id = int(box.cls.item())
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    all_detections.append({
                        "label": labels[cls_id],
                        "bbox": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0],
                        "conf": float(box.conf.item()),
                    })

        return self._dedup_by_iou(all_detections, iou_merge_thresh)

    def detect_visual_prompt(
        self,
        image: Union[Image.Image, str],
        ref_image: Union[Image.Image, str],
        ref_bboxes: List[List[float]],
        ref_label: str,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects matching example crops from a reference image rather
        than a text description - useful for categories that are awkward to
        phrase in words (a specific sign style, a specific flag design)
        where example crops describe the target better than text would.
        ref_bboxes are [x1, y1, x2, y2] pixel boxes drawn on ref_image around
        example instances of ref_label.
        Returns: list of dicts: {"label", "bbox", "conf"}
        """
        # yoloe-*-seg checkpoints return segmentation-shaped output, which
        # only YOLOEVPSegPredictor (not the plain detect variant) postprocesses correctly
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        if isinstance(ref_image, str):
            ref_img = Image.open(ref_image).convert("RGB")
        else:
            ref_img = ref_image.convert("RGB")

        visual_prompts = {
            "bboxes": np.array(ref_bboxes, dtype=np.float32),
            "cls": np.array([0] * len(ref_bboxes), dtype=np.int64),
        }
        results = self.model.predict(
            img,
            visual_prompts=visual_prompts,
            refer_image=ref_img,
            predictor=YOLOEVPSegPredictor,
            conf=DETECTION_CONF_THRESHOLD,
            verbose=False,
        )
        r = results[0]

        detections = []
        for box in r.boxes:
            detections.append({
                "label": ref_label,
                "bbox": box.xyxy[0].tolist(),
                "conf": float(box.conf.item())
            })
        return detections

    @staticmethod
    def _iou(box_a: List[float], box_b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _dedup_by_iou(self, detections: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
        """Greedy per-label NMS, highest confidence first - dedupes detections of the same object seen in overlapping tiles."""
        by_label = {}
        for det in detections:
            by_label.setdefault(det["label"], []).append(det)

        merged = []
        for dets in by_label.values():
            dets.sort(key=lambda d: d["conf"], reverse=True)
            kept = []
            for det in dets:
                if all(self._iou(det["bbox"], k["bbox"]) < iou_thresh for k in kept):
                    kept.append(det)
            merged.extend(kept)
        return merged

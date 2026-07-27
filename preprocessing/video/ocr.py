import unicodedata
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from paddleocr import PaddleOCR
from preprocessing.config import (
    OCR_LANG, OCR_REC_SCORE_THRESHOLD, OCR_REGION_CONCEPTS_EN,
    SAHI_TILE_SIZE, SAHI_TILE_OVERLAP, OCR_SR_MIN_HEIGHT_PX,
)

# Simple dictionary mapping for removing Vietnamese accents
ACCENT_MAP = {
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'đ': 'd',
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'ã': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
    'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
    'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
    'Đ': 'D',
    'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
    'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
    'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
    'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
    'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
    'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
    'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
    'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
    'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y'
}

def remove_vietnamese_accents(text: str) -> str:
    """
    Remove accents from Vietnamese text.
    """
    res = []
    for c in text:
        res.append(ACCENT_MAP.get(c, c))
    return "".join(res)

class TextDetectorOCR:
    """
    SAM3-gated OCR pipeline: a RegionProposer (SAM3, zero-shot concept
    segmentation) proposes candidate text/sign regions first - PP-OCRv6
    detection + SAHI-style tiling only run inside those regions, and OCR is
    skipped entirely for a keyframe where SAM3 finds no candidate region.
    Recognition on each surviving text-box crop is an ensemble of PP-OCRv6 +
    Vintern-1B-v3.5 (highest confidence wins), with small crops (< 16px tall)
    passed through Real-ESRGAN x4 first and a dedicated lightweight fallback
    VLM re-reading crops where the ensemble's best confidence is still below
    OCR_REC_SCORE_THRESHOLD.
    """
    def __init__(
        self,
        region_proposer,
        vintern,
        fallback_vlm,
        sr_model,
        lang: str = OCR_LANG,
    ):
        self.region_proposer = region_proposer
        self.vintern = vintern
        self.fallback_vlm = fallback_vlm
        self.sr_model = sr_model
        print(f"Loading PP-OCRv6 OCR model (lang={lang})...")
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print("PP-OCRv6 model loaded successfully.")

    def _detect_recognize(self, image: Image.Image, tile_origin=(0, 0)) -> List[Dict[str, Any]]:
        """Run PP-OCRv6 on a single image/tile, offsetting boxes back to full-frame coordinates."""
        img_np = np.array(image.convert("RGB"))
        results = self.ocr.predict(img_np)
        if not results:
            return []

        r = results[0]
        ox, oy = tile_origin
        boxes = []
        for text, score, box in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            boxes.append({
                "text": text,
                "confidence": float(score),
                "bbox": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
            })
        return boxes

    def _detect_recognize_tiled(
        self,
        image: Image.Image,
        tile_size: int = SAHI_TILE_SIZE,
        overlap: float = SAHI_TILE_OVERLAP,
        iou_merge_thresh: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Overlapping-tile pass over a (region-cropped) sub-image so small text
        surviving a full-region downscale isn't lost (mirrors
        ObjectDetector.detect_tiled in models/object_detector.py). Written by
        hand rather than via the `sahi` library's SAHI slicing helper: sahi
        0.12.1's model registry (sahi.auto_model.MODEL_TYPE_TO_MODEL_CLASS_NAME)
        has no "paddleocr" backend to plug into, so
        `AutoDetectionModel.from_pretrained(model_type="paddleocr", ...)`
        isn't runnable as-is.
        """
        width, height = image.size
        stride = max(int(tile_size * (1 - overlap)), 1)

        xs = list(range(0, max(width - tile_size, 0) + 1, stride)) or [0]
        ys = list(range(0, max(height - tile_size, 0) + 1, stride)) or [0]
        if xs[-1] + tile_size < width:
            xs.append(max(width - tile_size, 0))
        if ys[-1] + tile_size < height:
            ys.append(max(height - tile_size, 0))

        all_boxes = self._detect_recognize(image)
        for y0 in ys:
            for x0 in xs:
                x1, y1 = min(x0 + tile_size, width), min(y0 + tile_size, height)
                tile = image.crop((x0, y0, x1, y1))
                all_boxes.extend(self._detect_recognize(tile, tile_origin=(x0, y0)))

        return self._dedup_by_iou(all_boxes, iou_merge_thresh)

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

    def _dedup_by_iou(self, boxes: List[Dict[str, Any]], iou_thresh: float) -> List[Dict[str, Any]]:
        """Greedy NMS, highest confidence first - dedupes boxes seen in overlapping tiles/regions."""
        boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
        kept = []
        for box in boxes:
            if all(self._iou(box["bbox"], k["bbox"]) < iou_thresh for k in kept):
                kept.append(box)
        return kept

    def has_text(self, image: Image.Image) -> bool:
        """Cheap presence check reusing the same PP-OCRv6 detector (no separate model needed)."""
        return len(self._detect_recognize(image)) > 0

    def _propose_text_regions(self, image: Image.Image) -> List[List[float]]:
        """SAM3 pre-filter: candidate text/sign regions, or [] if none matched anywhere in the frame."""
        regions = self.region_proposer.propose(image, OCR_REGION_CONCEPTS_EN)
        return [r["bbox"] for r in regions]

    def _detect_text_boxes(self, image: Image.Image, region_bboxes: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Detection-only pass: SAHI-tiles within each SAM3-proposed region and
        keeps only the resulting bboxes - the tiled pass's own recognized
        text is discarded here since recognition happens as a separate
        ensemble stage in _recognize_crop, on the (possibly SR'd) final crop.
        """
        width, height = image.size
        all_boxes = []
        for rx1, ry1, rx2, ry2 in region_bboxes:
            rx1, ry1 = max(0, int(rx1)), max(0, int(ry1))
            rx2, ry2 = min(width, int(rx2)), min(height, int(ry2))
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            region_crop = image.crop((rx1, ry1, rx2, ry2))
            tiled = self._detect_recognize_tiled(region_crop)
            for box in tiled:
                bx1, by1, bx2, by2 = box["bbox"]
                box["bbox"] = [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1]
            all_boxes.extend(tiled)

        return self._dedup_by_iou(all_boxes, iou_thresh=0.3)

    def _recognize_crop(self, crop: Image.Image) -> Tuple[str, float, str]:
        """
        Conditional Super-Resolution + PP-OCRv6/Vintern-1B-v3.5 recognition
        ensemble + fallback-VLM escalation for a single text-box crop.
        Returns: (text, rec_conf, source)
        """
        if min(crop.size) == 0:
            return "", 0.0, "none"

        if crop.height < OCR_SR_MIN_HEIGHT_PX and self.sr_model is not None:
            try:
                crop = self.sr_model.upscale(crop)
            except Exception as e:
                print(f"Warning: Real-ESRGAN upscaling failed for a small OCR crop: {e}")

        pp_text, pp_conf = "", 0.0
        pp_candidates = self._detect_recognize(crop)
        if pp_candidates:
            best_pp = max(pp_candidates, key=lambda b: b["confidence"])
            pp_text, pp_conf = best_pp["text"], best_pp["confidence"]

        vintern_text, vintern_conf = "", 0.0
        if self.vintern is not None:
            try:
                vintern_text, vintern_conf = self.vintern.recognize(crop)
            except Exception as e:
                print(f"Warning: Vintern-1B-v3.5 recognition failed for an OCR crop: {e}")

        if vintern_conf > pp_conf:
            text, confidence, source = vintern_text, vintern_conf, "vintern-1b-v3.5"
        else:
            text, confidence, source = pp_text, pp_conf, "pp-ocrv6"

        if confidence < OCR_REC_SCORE_THRESHOLD and self.fallback_vlm is not None:
            try:
                escalated = self.fallback_vlm.generate(
                    crop, "Extract the text in this image exactly. Output only the text."
                ).strip()
                if escalated:
                    text, source = escalated, "fallback-vlm"
            except Exception as e:
                print(f"Warning: fallback VLM OCR escalation failed for a low-confidence crop: {e}")

        return text, confidence, source

    def extract_ocr_detailed(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Full SAM3-gated OCR pipeline for one keyframe. Returns a structured
        per-box list: [{"bbox", "text" (NFC-normalized), "conf",
        "accentless_text", "source"}], or [] if SAM3 found no candidate
        text/sign region (OCR is skipped entirely in that case).
        """
        region_bboxes = self._propose_text_regions(image)
        if not region_bboxes:
            return []

        text_boxes = self._detect_text_boxes(image, region_bboxes)

        results = []
        for box in text_boxes:
            x1, y1, x2, y2 = box["bbox"]
            crop = image.crop((max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)))
            text, confidence, source = self._recognize_crop(crop)
            if not text:
                continue

            normalized = unicodedata.normalize("NFC", text)
            results.append({
                "bbox": box["bbox"],
                "text": normalized,
                "conf": confidence,
                "accentless_text": remove_vietnamese_accents(normalized),
                "source": source,
            })

        return results

    @staticmethod
    def flatten_ocr_text(detailed_results: List[Dict[str, Any]]) -> str:
        """
        Pure helper (no model calls) - joins every box's accented + accentless
        text into one blob for BM25 dual-indexing. Call this on an
        extract_ocr_detailed() result instead of extract_ocr() when both the
        structured and flattened forms are needed, so the (expensive)
        pipeline only runs once per keyframe.
        """
        if not detailed_results:
            return ""
        accented = " ".join(r["text"] for r in detailed_results if r["text"])
        accentless = " ".join(r["accentless_text"] for r in detailed_results if r["accentless_text"])
        return f"{accented} {accentless}".strip()

    def extract_ocr(self, image: Image.Image) -> str:
        """Convenience single-call wrapper around extract_ocr_detailed() + flatten_ocr_text()."""
        return self.flatten_ocr_text(self.extract_ocr_detailed(image))

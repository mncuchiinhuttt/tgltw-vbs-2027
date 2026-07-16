import unicodedata
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from paddleocr import PaddleOCR
from preprocessing.config import OCR_LANG, OCR_REC_SCORE_THRESHOLD, OCR_USE_TILING

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

def normalize_vn_ocr(text: str) -> str:
    """
    Normalize Unicode to NFC form and index both accented and unaccented Vietnamese text.
    """
    normalized = unicodedata.normalize("NFC", text)
    no_accent = remove_vietnamese_accents(normalized)
    return f"{normalized} {no_accent}"

class TextDetectorOCR:
    """
    PP-OCRv6 text detection + recognition, replacing the previous path where
    every frame's OCR was delegated to the VLM. The VLM is now only used to
    re-read individual crops PP-OCRv6 recognized with low confidence, instead
    of running on every frame.
    """
    def __init__(self, vlm_client, lang: str = OCR_LANG, use_tiling: bool = OCR_USE_TILING):
        self.vlm = vlm_client
        self.use_tiling = use_tiling
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
        tile_size: int = 768,
        overlap: float = 0.2,
        iou_merge_thresh: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Supplementary overlapping-tile pass (mirrors ObjectDetector.detect_tiled
        in models/object_detector.py) so small or corner text surviving a
        full-frame downscale isn't lost. Written by hand rather than via the
        `sahi` library's SAHI slicing helper: sahi 0.12.1's model registry
        (sahi.auto_model.MODEL_TYPE_TO_MODEL_CLASS_NAME) has no "paddleocr"
        backend to plug into, so `AutoDetectionModel.from_pretrained(model_type="paddleocr", ...)`
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
        """Greedy NMS, highest confidence first - dedupes boxes seen in overlapping tiles."""
        boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
        kept = []
        for box in boxes:
            if all(self._iou(box["bbox"], k["bbox"]) < iou_thresh for k in kept):
                kept.append(box)
        return kept

    def has_text(self, image: Image.Image) -> bool:
        """Cheap presence check reusing the same PP-OCRv6 detector (no separate model needed)."""
        return len(self._detect_recognize(image)) > 0

    def extract_ocr_boxes(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect+recognize and return the raw per-box results ({bbox, text, confidence})."""
        if self.use_tiling:
            return self._detect_recognize_tiled(image)
        return self._detect_recognize(image)

    def extract_ocr(self, image: Image.Image) -> str:
        """
        Detect+recognize text via PP-OCRv6. Boxes recognized with confidence
        below OCR_REC_SCORE_THRESHOLD are escalated to the VLM to re-read
        just that crop - only a few blurry/hard cases hit the VLM, not the
        whole frame.
        """
        boxes = self.extract_ocr_boxes(image)
        if not boxes:
            return ""

        texts = []
        for box in boxes:
            text = box["text"]
            if box["confidence"] < OCR_REC_SCORE_THRESHOLD and self.vlm is not None:
                x1, y1, x2, y2 = box["bbox"]
                crop = image.crop((max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)))
                if min(crop.size) > 0:
                    try:
                        escalated = self.vlm.generate(
                            crop, "Extract the text in this image exactly. Output only the text."
                        ).strip()
                        if escalated:
                            text = escalated
                    except Exception as e:
                        print(f"Warning: VLM OCR escalation failed for a low-confidence crop: {e}")
            texts.append(text)

        return normalize_vn_ocr(" ".join(t for t in texts if t))

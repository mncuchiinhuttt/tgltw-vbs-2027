import unicodedata
import re
from PIL import Image
from typing import Union, List

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
    Text detector that checks if text is present in the frame (e.g. using CRAFT/DB-Net style check)
    and then triggers a VLM to extract the actual text.
    """
    def __init__(self, vlm_client):
        self.vlm = vlm_client
        # Placeholder for CRAFT or DB-Net. EasyOCR makes a good surrogate for local checking.
        self.local_detector = None
        try:
            import easyocr
            self.local_detector = easyocr.Reader(['vi', 'en'])
        except ImportError:
            print("easyocr is not installed. Will default to calling VLM for all frames.")

    def has_text(self, image: Image.Image) -> bool:
        """
        Check if the keyframe has any text using local detector (CRAFT / DB-Net / EasyOCR surrogate).
        """
        if self.local_detector is None:
            return True # Fallback: always run VLM
            
        import numpy as np
        img_np = np.array(image)
        results = self.local_detector.readtext(img_np)
        return len(results) > 0

    def extract_ocr(self, image: Image.Image) -> str:
        """
        Extract OCR text using VLM if text is detected in the keyframe.
        """
        if not self.has_text(image):
            return ""
            
        prompt = "Extract all text or signs visible on the screen. Return only the extracted text in Vietnamese/English, or nothing if there is no readable text."
        raw_text = self.vlm.generate(image, prompt)
        return normalize_vn_ocr(raw_text.strip())

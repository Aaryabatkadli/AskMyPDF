"""
Extracts text from a PDF, page by page. If a page has little/no
selectable text (i.e. it's a scanned image), falls back to Tesseract OCR.
"""
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

MIN_TEXT_CHARS_BEFORE_OCR = 20  # below this, treat page as scanned/image


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Returns a list of dicts:
    [{"page_number": 1, "text": "...", "used_ocr": False}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text").strip()
        used_ocr = False

        if len(text) < MIN_TEXT_CHARS_BEFORE_OCR:
            # Likely a scanned page -> render to image and OCR it
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(image)
            if len(ocr_text.strip()) > len(text):
                text = ocr_text.strip()
                used_ocr = True

        pages.append({
            "page_number": page_index + 1,
            "text": text,
            "used_ocr": used_ocr,
        })

    doc.close()
    return pages


def get_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count

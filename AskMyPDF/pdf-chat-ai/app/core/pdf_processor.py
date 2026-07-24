"""
Extracts text from a PDF, page by page. If a page has little/no
selectable text (i.e. it's a scanned image), falls back to Tesseract OCR.
If Tesseract isn't installed on the host (e.g. a deploy environment where
packages.txt wasn't picked up), OCR is skipped gracefully instead of
crashing - the page just keeps whatever selectable text it has (possibly
none, for a fully scanned page).
"""
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

MIN_TEXT_CHARS_BEFORE_OCR = 20  # below this, treat page as scanned/image

_ocr_warning_shown = False  # only warn once per process, not once per page


def _ocr_page_image(image) -> str:
    """Runs Tesseract OCR on an image. Returns '' if Tesseract isn't available."""
    global _ocr_warning_shown
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.pytesseract.TesseractNotFoundError:
        if not _ocr_warning_shown:
            print(
                "[pdf_processor] Tesseract is not installed on this host - "
                "OCR is disabled. Scanned/image-only pages will have no "
                "extractable text. Make sure packages.txt (containing "
                "'tesseract-ocr') sits in the same folder as your app's "
                "main entrypoint if deploying to Streamlit Cloud."
            )
            _ocr_warning_shown = True
        return ""


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
            # Likely a scanned page -> render to image and OCR it (if available)
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            ocr_text = _ocr_page_image(image)
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

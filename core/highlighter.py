"""
Renders a specific PDF page as an image, highlighting the region(s) that
match the retrieved chunk text (best-effort search on the page for the
first ~80 characters of the chunk).
"""
import fitz  # PyMuPDF
from PIL import Image
import io


def render_highlighted_page(pdf_path: str, page_number: int, search_text: str) -> bytes:
    """
    page_number is 1-indexed. Returns PNG bytes of the rendered page,
    with any matches of a snippet of search_text highlighted in yellow.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]

    snippet = search_text.strip()[:80]
    if snippet:
        rects = page.search_for(snippet, quads=False)
        for rect in rects:
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 1, 0))
            highlight.update()

    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def render_plain_page(pdf_path: str, page_number: int) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes

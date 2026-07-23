"""
Splits extracted page text into overlapping chunks, keeping track of
which page (and roughly which position) each chunk came from so we can
cite sources and highlight later.
"""
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: list[dict], doc_name: str) -> list[dict]:
    """
    pages: output of pdf_processor.extract_pages
    Returns a list of chunk dicts:
    [{"id": "...", "text": "...", "page_number": 1, "doc_name": "file.pdf"}]
    """
    chunks = []
    chunk_counter = 0

    for page in pages:
        text = page["text"]
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_counter += 1
                chunks.append({
                    "id": f"{doc_name}::p{page['page_number']}::c{chunk_counter}",
                    "text": chunk_text,
                    "page_number": page["page_number"],
                    "doc_name": doc_name,
                })

            if end == len(text):
                break
            start = end - CHUNK_OVERLAP  # overlap for context continuity

    return chunks

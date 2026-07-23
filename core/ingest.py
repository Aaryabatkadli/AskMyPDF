"""
Orchestrates the full ingestion pipeline for one uploaded PDF:
extract text -> chunk -> embed -> store in ChromaDB.
"""
from app.core.pdf_processor import extract_pages
from app.core.chunker import chunk_pages
from app.core.vector_store import VectorStore


def ingest_pdf(pdf_path: str, doc_name: str, store: VectorStore) -> dict:
    """
    Returns a summary dict: {"pages": int, "chunks": int, "ocr_pages": int}
    """
    # Avoid re-indexing the same document if it's already in the store
    if store.has_document(doc_name):
        return {"pages": None, "chunks": 0, "ocr_pages": 0, "already_indexed": True}

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, doc_name)
    store.add_chunks(chunks)

    ocr_pages = sum(1 for p in pages if p["used_ocr"])

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "ocr_pages": ocr_pages,
        "already_indexed": False,
    }

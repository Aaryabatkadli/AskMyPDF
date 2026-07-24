"""
Wraps ChromaDB (persistent, local, free) using fastembed for embeddings.

fastembed runs on ONNX Runtime instead of PyTorch, which keeps memory
usage far lower than sentence-transformers - important for running
comfortably within Streamlit Community Cloud's free-tier resource limits.
"""
import chromadb
from fastembed import TextEmbedding
from app.config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K

_embedder = None


class EmbeddingError(Exception):
    pass


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        try:
            _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        except Exception as e:
            raise EmbeddingError(
                f"Couldn't load the embedding model ({EMBEDDING_MODEL}): {e}"
            )
    return _embedder


def _embed(embedder: TextEmbedding, texts: list[str]) -> list[list[float]]:
    # fastembed returns a generator of numpy arrays - materialize to plain lists
    return [vec.tolist() for vec in embedder.embed(texts)]


class VectorStore:
    def __init__(self, collection_name: str = "pdf_chat"):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list[dict]):
        if not chunks:
            return
        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        try:
            embeddings = _embed(embedder, texts)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings for this document: {e}")

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"page_number": c["page_number"], "doc_name": c["doc_name"]}
                for c in chunks
            ],
        )

    def query(self, question: str, doc_name: str | None = None, top_k: int = TOP_K) -> list[dict]:
        embedder = get_embedder()
        query_embedding = _embed(embedder, [question])

        where = {"doc_name": doc_name} if doc_name else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )

        matches = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            # Chroma returns a distance (lower = more similar).
            # Convert to a rough 0-100% confidence score for display.
            similarity = max(0.0, 1 - dist)
            matches.append({
                "text": doc,
                "page_number": meta.get("page_number"),
                "doc_name": meta.get("doc_name"),
                "confidence": round(similarity * 100, 1),
            })

        return matches

    def clear_document(self, doc_name: str):
        self.collection.delete(where={"doc_name": doc_name})

    def has_document(self, doc_name: str) -> bool:
        existing = self.collection.get(where={"doc_name": doc_name}, limit=1)
        return len(existing.get("ids", [])) > 0

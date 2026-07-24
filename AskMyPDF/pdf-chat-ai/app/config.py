"""
Central configuration for the app. Reads from environment variables (.env)
locally, and from Streamlit secrets when deployed on Streamlit Community Cloud.
"""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}


def _get(key: str, default: str = "") -> str:
    """Checks Streamlit secrets first (for cloud deploy), then env vars (for local).
    Safe to call even when no secrets.toml exists at all (e.g. local dev)."""
    try:
        if key in _SECRETS:
            return str(_SECRETS[key])
    except Exception:
        pass  # no secrets.toml present locally - that's fine, fall back to .env
    return os.getenv(key, default)


# --- LLM provider switch ---
# "ollama" -> runs 100% locally, no API key, used for local development.
# "groq"   -> free-tier cloud LLM, used for the deployed Streamlit Cloud app
#             (Ollama cannot run on Streamlit Cloud's servers).
LLM_PROVIDER = _get("LLM_PROVIDER", "ollama").lower()

# --- Ollama (local LLM, no API key) ---
OLLAMA_HOST = _get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.2")

# --- Groq (free-tier cloud LLM, used only when LLM_PROVIDER=groq) ---
GROQ_API_KEY = _get("GROQ_API_KEY", "")
GROQ_MODEL = _get("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Embeddings (local, downloaded once from HuggingFace, free) ---
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Chunking ---
CHUNK_SIZE = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
TOP_K = int(_get("TOP_K", "4"))

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

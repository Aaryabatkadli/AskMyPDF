"""
Answers questions using retrieved chunks as context, via either:
  - Ollama (local, no API key)     -> used when LLM_PROVIDER=ollama (local dev)
  - Groq  (free-tier cloud API)    -> used when LLM_PROVIDER=groq   (Streamlit Cloud deploy)

The rest of the app never needs to know which one is active - it just
calls ask_question() and check_llm_available().
"""
import requests
from app.config import (
    LLM_PROVIDER,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided document excerpts. If the answer isn't in the excerpts, say "
    "you don't know based on the document. Always be concise and accurate. "
    "When relevant, mention which page the information came from."
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMConnectionError(Exception):
    pass


# Kept as an alias so any older imports of OllamaConnectionError still work.
OllamaConnectionError = LLMConnectionError


def build_context(matches: list[dict]) -> str:
    parts = []
    for m in matches:
        parts.append(f"[Page {m['page_number']}]\n{m['text']}")
    return "\n\n---\n\n".join(parts)


def build_user_prompt(question: str, matches: list[dict]) -> str:
    context = build_context(matches)
    return (
        f"Document excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above."
    )


# ---------------------------------------------------------------------------
# Ollama backend (local dev)
# ---------------------------------------------------------------------------
def _check_ollama_available() -> bool:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _ask_ollama(question: str, matches: list[dict], model: str = None) -> str:
    if not _check_ollama_available():
        raise LLMConnectionError(
            f"Can't reach Ollama at {OLLAMA_HOST}. Make sure Ollama is installed "
            "and running ('ollama serve' or the Ollama app), and that you've "
            f"pulled a model with 'ollama pull {OLLAMA_MODEL}'."
        )

    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, matches)},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise LLMConnectionError(f"Ollama request failed: {e}")

    return resp.json().get("message", {}).get("content", "").strip()


def _list_ollama_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except requests.exceptions.RequestException:
        return []


# ---------------------------------------------------------------------------
# Groq backend (free-tier cloud, used for the deployed app)
# ---------------------------------------------------------------------------
def _check_groq_available() -> bool:
    return bool(GROQ_API_KEY)


def _ask_groq(question: str, matches: list[dict], model: str = None) -> str:
    if not GROQ_API_KEY:
        raise LLMConnectionError(
            "No Groq API key configured. Add GROQ_API_KEY in your Streamlit Cloud "
            "app's Secrets (Settings -> Secrets), or in .env for local testing. "
            "Get a free key at https://console.groq.com/keys"
        )

    payload = {
        "model": model or GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, matches)},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            pass
        raise LLMConnectionError(f"Groq request failed: {detail or e}")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Public, provider-agnostic API used by main.py
# ---------------------------------------------------------------------------
def check_llm_available() -> bool:
    if LLM_PROVIDER == "groq":
        return _check_groq_available()
    return _check_ollama_available()


# Backward-compatible name used by the earlier local-only version of the app
def check_ollama_available() -> bool:
    return check_llm_available()


def list_available_models() -> list[str]:
    if LLM_PROVIDER == "groq":
        return [GROQ_MODEL]
    return _list_ollama_models()


def ask_question(question: str, matches: list[dict], model: str = None) -> str:
    if LLM_PROVIDER == "groq":
        return _ask_groq(question, matches, model)
    return _ask_ollama(question, matches, model)

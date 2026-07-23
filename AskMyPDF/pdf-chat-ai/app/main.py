"""
AskMyPDF - RAG-based PDF chat assistant.

Supports two LLM backends via LLM_PROVIDER in config:
  - "ollama" : 100% local, no API key - used for local development
  - "groq"   : free-tier cloud LLM - used for the public Streamlit Cloud deployment
"""
import os
import sys

# Allow running via `streamlit run app/main.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.config import UPLOAD_DIR, OLLAMA_MODEL, LLM_PROVIDER
from app.core.ingest import ingest_pdf
from app.core.vector_store import VectorStore
from app.core.qa_engine import ask_question, check_llm_available, list_available_models, LLMConnectionError
from app.core.highlighter import render_highlighted_page

st.set_page_config(page_title="AskMyPDF", page_icon="📄", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Overall app background - deep navy with soft glow, like the AI Chat Pro kit */
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(56, 189, 248, 0.10) 0%, transparent 45%),
            radial-gradient(circle at 85% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
            linear-gradient(180deg, #060a17 0%, #0a1024 55%, #0a0e1f 100%);
    }

    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Inter, sans-serif;
    }

    /* Hero header - glowing blue/cyan gradient banner */
    .askmypdf-hero {
        position: relative;
        padding: 2rem 2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #0b2447 0%, #14589e 45%, #19a7ce 100%);
        margin-bottom: 1.8rem;
        box-shadow: 0 0 0 1px rgba(94, 197, 255, 0.25), 0 10px 40px rgba(20, 88, 158, 0.45);
        overflow: hidden;
    }
    .askmypdf-hero::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -10%;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(120, 220, 255, 0.35) 0%, transparent 70%);
        pointer-events: none;
    }
    .askmypdf-hero h1 {
        color: white;
        font-size: 2.1rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-shadow: 0 0 24px rgba(120, 220, 255, 0.5);
    }
    .askmypdf-hero p {
        color: rgba(220, 245, 255, 0.9);
        margin: 0.35rem 0 0 0;
        font-size: 0.98rem;
    }
    .askmypdf-badge {
        display: inline-block;
        margin-top: 0.9rem;
        padding: 5px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        color: #e0f7ff;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #070b18;
        border-right: 1px solid rgba(94, 197, 255, 0.12);
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #e5f2ff;
    }

    /* New Chat button - glowing cyan */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(120deg, #14589e 0%, #19a7ce 100%) !important;
        border: 1px solid rgba(94, 197, 255, 0.4) !important;
        box-shadow: 0 0 18px rgba(25, 167, 206, 0.45);
        font-weight: 700 !important;
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .status-online {
        background: rgba(25, 167, 206, 0.15);
        color: #5ec5ff;
        border: 1px solid rgba(25, 167, 206, 0.4);
        box-shadow: 0 0 12px rgba(25, 167, 206, 0.25);
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Confidence badges */
    .conf-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .conf-high { background: rgba(25,167,206,0.18); color: #5ec5ff; }
    .conf-mid  { background: rgba(234,179,8,0.18); color: #facc15; }
    .conf-low  { background: rgba(239,68,68,0.18); color: #f87171; }

    /* History cards in sidebar */
    .history-card {
        background: rgba(94, 197, 255, 0.05);
        border: 1px solid rgba(94, 197, 255, 0.12);
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        color: #cfe9ff;
        cursor: default;
    }
    .history-card b { color: #f3f4f6; }

    /* Upload dropzone container - glowing cyan dashed border */
    .upload-shell {
        border: 1.5px dashed rgba(25, 167, 206, 0.45);
        border-radius: 16px;
        padding: 0.5rem 0.9rem 1rem 0.9rem;
        background: rgba(25, 167, 206, 0.05);
        box-shadow: inset 0 0 24px rgba(25, 167, 206, 0.06);
        margin-bottom: 1.2rem;
    }

    /* Section divider label */
    .section-label {
        color: #7fb8dd;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        margin: 1rem 0 0.4rem 0;
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        background: rgba(94, 197, 255, 0.05);
        border: 1px solid rgba(94, 197, 255, 0.10);
        border-radius: 14px;
        padding: 0.4rem 0.2rem;
    }

    /* Buttons generally */
    button {
        border-radius: 10px !important;
    }

    footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_store():
    return VectorStore()


def confidence_badge(confidence: float) -> str:
    if confidence >= 70:
        cls = "conf-high"
    elif confidence >= 40:
        cls = "conf-mid"
    else:
        cls = "conf-low"
    return f'<span class="conf-badge {cls}">{confidence}% match</span>'


def render_sources(matches, pdf_path):
    with st.expander("📎 Sources & confidence", expanded=False):
        for src in matches:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Page {src['page_number']}**", unsafe_allow_html=True)
                st.markdown(confidence_badge(src["confidence"]), unsafe_allow_html=True)
            with col2:
                img_bytes = render_highlighted_page(pdf_path, src["page_number"], src["text"])
                st.image(img_bytes, use_container_width=True)


def main():
    # --- Hero header ---
    st.markdown(
        """
        <div class="askmypdf-hero">
            <h1>🤖 AskMyPDF</h1>
            <p>Chat with your documents — fully local, private, and free.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    store = get_store()

    # --- Session state ---
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "pdf_paths" not in st.session_state:
        st.session_state.pdf_paths = {}          # doc_name -> file path, remembered for review
    if "history_log" not in st.session_state:
        st.session_state.history_log = []        # every Q&A ever asked, across all PDFs

    # --- Sidebar: New chat + Ollama status + model picker + chat history ---
    with st.sidebar:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.doc_name = None
            st.session_state.pdf_path = None
            st.rerun()

        st.divider()

        st.header("⚙️ Settings")
        llm_ok = check_llm_available()

        if LLM_PROVIDER == "groq":
            # Public deployment: keep it simple, no backend jargon, no model picker.
            if llm_ok:
                st.markdown(
                    '<div class="status-pill status-online">🟢 AI Engine Online</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="status-pill status-offline">🔴 AI Engine Offline</div>',
                    unsafe_allow_html=True,
                )
                st.caption("The assistant is temporarily unavailable. Please try again shortly.")
            selected_model = None  # qa_engine uses the configured default automatically

        else:
            # Local development: show Ollama status + let the user pick a pulled model.
            if llm_ok:
                st.markdown(
                    '<div class="status-pill status-online">🟢 Ollama running</div>',
                    unsafe_allow_html=True,
                )
                models = list_available_models()
                if models:
                    selected_model = st.selectbox("Model", models, index=(
                        models.index(OLLAMA_MODEL) if OLLAMA_MODEL in models else 0
                    ))
                else:
                    st.warning(f"No models pulled yet. Run: `ollama pull {OLLAMA_MODEL}`")
                    selected_model = OLLAMA_MODEL
            else:
                st.markdown(
                    '<div class="status-pill status-offline">🔴 Ollama not detected</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "Install Ollama from [ollama.com](https://ollama.com), then run:\n\n"
                    f"```\nollama pull {OLLAMA_MODEL}\n```\n\n"
                    "Ollama usually starts automatically after install. "
                    "If not, run `ollama serve`."
                )
                selected_model = OLLAMA_MODEL

        # --- Chat history, directly below the model picker ---
        # Grouped by PDF, so the user can review every document they've
        # uploaded and every question/answer they had on it - permanently,
        # even after uploading other PDFs or starting a New Chat.
        st.markdown('<div class="section-label">💬 Chat History</div>', unsafe_allow_html=True)

        if st.session_state.history_log:
            # Group entries by document, most recently used document first
            docs_in_order = []
            for entry in st.session_state.history_log:
                if entry["doc_name"] not in docs_in_order:
                    docs_in_order.append(entry["doc_name"])
            docs_in_order.reverse()

            for doc in docs_in_order:
                doc_entries = [e for e in st.session_state.history_log if e["doc_name"] == doc]
                is_active = (doc == st.session_state.doc_name)
                label = f"{'📄' if not is_active else '📌'} {doc}  ({len(doc_entries)} Q&A)"

                with st.expander(label, expanded=False):
                    if not is_active:
                        if st.button("↩️ Resume this PDF", key=f"resume-{doc}", use_container_width=True):
                            st.session_state.doc_name = doc
                            st.session_state.pdf_path = st.session_state.pdf_paths.get(doc)
                            st.rerun()
                    for i, entry in enumerate(doc_entries, start=1):
                        st.markdown(f"**Q{i}:** {entry['question']}")
                        st.markdown(f"**A{i}:** {entry['answer']}")
                        st.markdown("---")
        else:
            st.caption("No questions asked yet. Every PDF and Q&A you have will be listed here.")

    # --- Upload ---
    st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload a PDF to get started", type=["pdf"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        doc_name = uploaded_file.name
        pdf_path = os.path.join(UPLOAD_DIR, doc_name)

        if st.session_state.doc_name != doc_name:
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Reading and indexing your PDF (OCR runs automatically on scanned pages)..."):
                summary = ingest_pdf(pdf_path, doc_name, store)

            st.session_state.doc_name = doc_name
            st.session_state.pdf_path = pdf_path
            st.session_state.pdf_paths[doc_name] = pdf_path

            if summary["already_indexed"]:
                st.info("This document was already indexed. Ready to chat!")
            else:
                st.success(
                    f"Indexed {summary['pages']} pages into {summary['chunks']} chunks "
                    f"({summary['ocr_pages']} pages used OCR)."
                )

    st.divider()

    # --- Chat ---
    if st.session_state.doc_name:
        st.subheader(f"💬 Ask questions about: {st.session_state.doc_name}")

        # Only show the conversation for the currently active document
        current_entries = [
            e for e in st.session_state.history_log if e["doc_name"] == st.session_state.doc_name
        ]

        for entry in current_entries:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])
                if entry.get("sources"):
                    render_sources(entry["sources"], st.session_state.pdf_path)

        question = st.chat_input("Ask something about the document...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            matches = store.query(question, doc_name=st.session_state.doc_name)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(question, matches, model=selected_model)
                    except LLMConnectionError as e:
                        answer = f"⚠️ {e}"
                st.write(answer)

                if matches:
                    render_sources(matches, st.session_state.pdf_path)

            st.session_state.history_log.append({
                "doc_name": st.session_state.doc_name,
                "question": question,
                "answer": answer,
                "sources": matches,
            })
            st.rerun()
    else:
        st.info("👆 Upload a PDF above to start chatting with it.")


if __name__ == "__main__":
    main()

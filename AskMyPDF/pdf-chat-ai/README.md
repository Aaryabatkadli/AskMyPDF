# 📄 AI PDF Chat — Local RAG App (Ollama, 100% Free)

Chat with any PDF using a fully local RAG (Retrieval-Augmented Generation) pipeline.
No API keys. No cloud costs. Everything — the LLM, the embeddings, and the vector
database — runs on your own machine.

## How it works

```
PDF upload
   ↓
Text extraction (PyMuPDF) — with Tesseract OCR fallback for scanned pages
   ↓
Chunking (overlapping text chunks, tagged with page number)
   ↓
Embedding (sentence-transformers, local)
   ↓
ChromaDB (local vector store)
   ↓
Similarity search on your question
   ↓
Ollama (local LLM) generates the answer from retrieved chunks
   ↓
Answer + page citation + confidence score + highlighted source page image
```

## Stack

- **UI:** Streamlit
- **PDF parsing:** PyMuPDF
- **OCR (scanned PDFs):** Tesseract via pytesseract
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`, downloads once, free)
- **Vector DB:** ChromaDB (local, persistent)
- **LLM:** Ollama (local, free, no API key) — default model `llama3.2`

## Prerequisites

1. **Python 3.10+**
2. **Tesseract OCR** (system package, not just the Python wrapper)
   - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH
   - Mac: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`
3. **Ollama** — install from https://ollama.com
   - After installing, pull a model:
     ```
     ollama pull llama3.2
     ```
   - Ollama runs a local server automatically (`http://localhost:11434`). If it's not
     running, start it manually with `ollama serve`.

## Setup (Windows PowerShell)

```powershell
# 1. Clone your repo (after you've pushed it, or just work in this folder)
cd pdf-chat-ai

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Copy the env file and adjust if needed
Copy-Item .env.example .env

# 5. Make sure Ollama is running and you've pulled a model
ollama pull llama3.2

# 6. Run the app
streamlit run app/main.py
```

## Setup (Mac/Linux)

```bash
cd pdf-chat-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2
streamlit run app/main.py
```

The app opens at `http://localhost:8501`.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` for local dev, `groq` for the deployed cloud app |
| `OLLAMA_HOST` | `http://localhost:11434` | Where your local Ollama server is running |
| `OLLAMA_MODEL` | `llama3.2` | Default model to use when `LLM_PROVIDER=ollama` |
| `GROQ_API_KEY` | *(empty)* | Free API key from console.groq.com, used when `LLM_PROVIDER=groq` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use when `LLM_PROVIDER=groq` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (HuggingFace) |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `4` | Number of chunks retrieved per question |

Locally, the sidebar lets you switch between any Ollama models you've pulled.
On the deployed (Groq) version, the model is fixed by `GROQ_MODEL` and the
sidebar simply shows whether the AI engine is online.

## Deploying to Streamlit Community Cloud (public link)

Ollama itself can't run on Streamlit Cloud's free servers, so the deployed
version of this app uses **Groq's free-tier API** instead — same RAG pipeline,
same UI, just a different LLM backend under the hood via the `LLM_PROVIDER`
switch. Locally you can keep using Ollama (free, no key, fully private);
your deployed public link uses Groq (free tier, requires a key but no cost).

### 1. Get a free Groq API key
Sign up at https://console.groq.com/keys and create a key. Groq's free tier
is generous and requires no credit card.

### 2. Push this repo to GitHub
```powershell
cd pdf-chat-ai
git init
git add .
git commit -m "Initial commit: local RAG PDF chat, Ollama + Groq"
git branch -M main
git remote add origin https://github.com/<your-username>/pdf-chat-ai.git
git push -u origin main
```

### 3. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, pick your repo/branch, and set the main file path to
   `app/main.py`.
3. Before (or right after) deploying, open **Settings -> Secrets** on the app
   and paste:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your-groq-api-key-here"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
4. Deploy. Streamlit Cloud will read `requirements.txt` and `packages.txt`
   (for the `tesseract-ocr` system package) automatically.

Your live link will now answer questions via Groq, while your local copy
(with `LLM_PROVIDER=ollama` in `.env`) keeps running fully offline.

### Note on data persistence
Streamlit Community Cloud's filesystem is ephemeral — uploaded PDFs and the
ChromaDB index reset whenever the app restarts or redeploys. That's expected
behavior for this kind of free-tier demo; each visitor just re-uploads their
PDF for their session.


## Project structure

```
pdf-chat-ai/
├── app/
│   ├── main.py                # Streamlit UI
│   ├── config.py              # Reads .env settings
│   └── core/
│       ├── pdf_processor.py   # Text extraction + OCR fallback
│       ├── chunker.py         # Splits text into overlapping chunks
│       ├── vector_store.py    # ChromaDB + sentence-transformers
│       ├── qa_engine.py       # Calls local Ollama server
│       ├── highlighter.py     # Renders highlighted source page images
│       └── ingest.py          # Orchestrates the ingestion pipeline
├── data/
│   ├── uploads/                # Uploaded PDFs (gitignored)
│   └── chroma_db/              # Vector DB storage (gitignored)
├── requirements.txt
├── packages.txt                 # System packages (tesseract) — for cloud deploy
├── .env.example
└── .gitignore
```

## Pushing to GitHub

```powershell
cd pdf-chat-ai
git init
git add .
git commit -m "Initial commit: local RAG PDF chat with Ollama"
git branch -M main
git remote add origin https://github.com/<your-username>/pdf-chat-ai.git
git push -u origin main
```

Make sure `.env` is **not** committed (it's already in `.gitignore`) — only
`.env.example` should be pushed, since it contains no secrets anyway (Ollama
needs no API key).

## Troubleshooting

- **"Ollama not detected"** → make sure Ollama is installed and running
  (`ollama serve`, or just open the Ollama desktop app). Check
  `http://localhost:11434` is reachable in your browser.
- **No models in dropdown** → run `ollama pull llama3.2` (or any other model,
  e.g. `ollama pull mistral`), then refresh the Streamlit page.
- **OCR not working / `TesseractNotFoundError`** → Tesseract must be installed as a
  system binary, not just via `pip install pytesseract`. See Prerequisites above.
- **Slow first run** → the embedding model downloads once (~80MB) the first time
  you run the app; subsequent runs are fast.

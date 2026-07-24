# Free RAG Chat
# Hugging Face Spaces (Streamlit) + local / Streamlit Community Cloud

Production-ready **Retrieval-Augmented Generation** chatbot in Python — **100% free** to run.

| Component | Technology | Cost |
|-----------|------------|------|
| LLM | [Groq](https://console.groq.com) `llama-3.3-70b-versatile` | Free API tier |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local CPU) | Free |
| Vector DB | ChromaDB (persistent on disk) | Free |
| UI | Streamlit chat | Free |
| Orchestration | LangChain | Free |

Documents are loaded from a folder you choose — typically a **Google Drive for desktop** sync directory, or the local `data/docs/` folder.

> **Note:** Vercel is not supported (Streamlit + local Chroma need a persistent server). Use **Streamlit Community Cloud** or **Hugging Face Spaces** instead.

---

## Features

- Load **PDF**, **TXT**, and **Markdown** from `DOCS_FOLDER_PATH`
- Chunk with `RecursiveCharacterTextSplitter` (1000 / 200 overlap)
- Persistent Chroma index under `data/vectorstore/`
- ChatGPT-style UI (`st.chat_input` / `st.chat_message`) with session history
- **Re-index / Sync** button with live progress
- Sidebar shows chunk count and source-file stats
- Answers grounded in retrieved context only, with **source citations** and expandable snippets
- Explicit “not found” response when the index has no answer

---

## Project layout

```
rag/
├── app.py                 # Streamlit chat UI
├── requirements.txt
├── .env.example
├── README.md
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── data/
│   ├── docs/              # Default document folder (committed sample included)
│   └── vectorstore/       # Persistent ChromaDB (created on sync, gitignored)
└── src/
    ├── config.py          # Env / Streamlit secrets / paths / model names
    ├── embeddings.py      # Local HuggingFace embeddings
    ├── document_loader.py # DirectoryLoader + splitters
    ├── vectorstore.py     # Chroma helpers
    ├── indexer.py         # Re-index / sync pipeline
    └── rag_chain.py       # Groq RAG Q&A
```

---

## Deploy (free)

### Option A — Streamlit Community Cloud (recommended)

1. Push this repo to GitHub (already done if you followed setup below).
2. Open [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select this repository → Main file path: `app.py`.
4. Under **Advanced settings → Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
DOCS_FOLDER_PATH = "data/docs"
```

5. Deploy. After the app boots, click **Re-index / Sync Documents** in the sidebar.

On the free tier the filesystem can reset between sleeps — re-sync after a cold start if the chunk count shows empty.

### Option B — Hugging Face Spaces

1. Create a new Space: [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. SDK: **Streamlit**, hardware: CPU basic (free).
3. Push/connect this GitHub repo, or upload the project files (`app.py` as the entrypoint).
4. In **Settings → Secrets**, add `GROQ_API_KEY` (and optionally `DOCS_FOLDER_PATH=data/docs`).
5. Wait for the build, then open the Space URL and run **Re-index / Sync Documents**.

---

## 1. Prerequisites

- Python **3.10+**
- A free [Groq](https://console.groq.com) account
- (Optional) [Google Drive for desktop](https://www.google.com/drive/download/) if you want Drive-synced docs

---

## 2. Get a free Groq API key

1. Open [https://console.groq.com](https://console.groq.com) and sign up / log in.
2. Go to **API Keys**: [https://console.groq.com/keys](https://console.groq.com/keys).
3. Click **Create API Key**, name it (e.g. `rag-local`), and copy the key.
4. Keep the key private — never commit it to git.

Groq’s free tier is enough for personal RAG use (rate limits apply).

---

## 3. Install

```bash
# From the project root
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the embedding model (`~90 MB`) via `sentence-transformers`.

---

## 4. Configure `.env`

```bash
# Windows
copy .env.example .env

# macOS / Linux
# cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_actual_key_here
DOCS_FOLDER_PATH=data/docs
```

### Pointing at Google Drive

Install **Google Drive for desktop** so Drive appears as a local folder, then set:

```env
# Windows example
DOCS_FOLDER_PATH=G:/My Drive/RAG_Docs

# macOS example
# DOCS_FOLDER_PATH=/Users/you/Library/CloudStorage/GoogleDrive-you@email/My Drive/RAG_Docs
```

Put PDF / TXT / MD files in that folder (or in `data/docs/`). Subfolders are included recursively.

---

## 5. Add documents & launch

1. Drop files into `DOCS_FOLDER_PATH`.
2. Start the app:

```bash
streamlit run app.py
```

3. Open the URL Streamlit prints (usually `http://localhost:8501`).
4. In the **sidebar**, click **Re-index / Sync Documents**.
5. Wait for the progress bar to finish — the chunk count should become non-zero.
6. Ask questions in the chat box.

---

## How the pipeline works

```
DOCS_FOLDER_PATH  →  DirectoryLoader (PDF/TXT/MD)
                  →  RecursiveCharacterTextSplitter
                  →  HuggingFaceEmbeddings (local)
                  →  ChromaDB (data/vectorstore/)
                  →  similarity retrieve (top-k)
                  →  ChatGroq (llama-3.3-70b-versatile)
                  →  Streamlit chat + source expanders
```

The system prompt forces the model to answer **only** from retrieved context and to cite source filenames. If nothing relevant is retrieved, the UI / model states that clearly.

---

## Optional environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `GROQ_API_KEY` | _(required)_ | Groq API key |
| `DOCS_FOLDER_PATH` | `data/docs` | Document root |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` | Chat model |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedder |
| `CHUNK_SIZE` | `1000` | Splitter chunk size |
| `CHUNK_OVERLAP` | `200` | Splitter overlap |
| `RETRIEVER_K` | `4` | Chunks retrieved per question |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `GROQ_API_KEY is not set` | Local: `.env`. Cloud: app Secrets UI |
| Empty vector DB after sync | Confirm files are `.pdf` / `.txt` / `.md` under `DOCS_FOLDER_PATH` |
| Drive path not found | Open the folder in Explorer/Finder; use the exact synced path |
| Slow first answer | Embedding model is downloading / loading into memory once |
| Import errors | Activate the venv and re-run `pip install -r requirements.txt` |
| Cloud app empty after sleep | Click **Re-index / Sync Documents** again |

---

## License

MIT — use freely for personal and commercial projects.

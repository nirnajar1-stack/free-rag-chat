"""
Free RAG Chat — Streamlit UI

ChatGPT-style interface over a local ChromaDB index, Groq LLM,
and HuggingFace embeddings. Documents are loaded from DOCS_FOLDER_PATH
(Google Drive sync folder or local data/docs/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path when launched via `streamlit run app.py`
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DOCS_FOLDER_PATH, EMBEDDING_MODEL_NAME, GROQ_MODEL_NAME, VECTORSTORE_PATH
from src.document_loader import count_source_files
from src.indexer import reindex_documents
from src.rag_chain import ask_question
from src.vectorstore import get_chunk_count

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Free RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lightweight styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 900px; }
    [data-testid="stSidebar"] { min-width: 320px; }
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #e8f0fe;
        color: #1a73e8;
    }
    .status-pill.empty { background: #fce8e6; color: #c5221f; }
    .status-pill.ready { background: #e6f4ea; color: #137333; }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_index_message" not in st.session_state:
        st.session_state.last_index_message = None


_init_session()


# ---------------------------------------------------------------------------
# Sidebar — sync & status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Free RAG")
    st.caption("LangChain · Groq · HuggingFace · ChromaDB")

    st.divider()
    st.subheader("Documents")
    st.code(str(DOCS_FOLDER_PATH), language=None)

    file_counts = count_source_files()
    st.markdown(
        f"**Source files:** {file_counts['total']}  \n"
        f"PDF: {file_counts.get('.pdf', 0)} · "
        f"TXT: {file_counts.get('.txt', 0)} · "
        f"MD: {file_counts.get('.md', 0) + file_counts.get('.markdown', 0)}"
    )

    st.divider()
    st.subheader("Vector database")
    chunk_count = get_chunk_count()
    pill_class = "ready" if chunk_count > 0 else "empty"
    label = f"{chunk_count} chunk(s) indexed" if chunk_count > 0 else "Empty — sync required"
    st.markdown(
        f'<span class="status-pill {pill_class}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Persist path: `{VECTORSTORE_PATH}`")

    st.divider()
    st.subheader("Sync")
    st.caption(
        "Pulls PDF / TXT / Markdown from the folder above "
        "(e.g. a Google Drive sync directory) and rebuilds the index."
    )

    if st.button("🔄 Re-index / Sync Documents", type="primary", use_container_width=True):
        status = st.empty()
        progress = st.progress(0, text="Starting sync…")

        steps = {"n": 0}

        def on_progress(msg: str) -> None:
            steps["n"] += 1
            # Soft progress — we don't know total steps ahead of time
            pct = min(95, 10 + steps["n"] * 8)
            progress.progress(pct, text=msg)
            status.info(msg)

        with st.spinner("Indexing documents…"):
            result = reindex_documents(progress_callback=on_progress)

        if result.success:
            progress.progress(100, text="Done")
            status.success(result.message)
            st.session_state.last_index_message = result.message
            st.cache_resource.clear()
            st.rerun()
        else:
            progress.progress(100, text="Failed")
            status.error(result.message)
            st.session_state.last_index_message = result.message

    if st.session_state.last_index_message:
        st.caption(f"Last sync: {st.session_state.last_index_message}")

    st.divider()
    st.subheader("Models")
    st.markdown(
        f"- **LLM:** `{GROQ_MODEL_NAME}` (Groq)\n"
        f"- **Embeddings:** `{EMBEDDING_MODEL_NAME}` (local)"
    )

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Ask your documents")
st.markdown(
    "Answers are grounded in your indexed files only. "
    "Sources appear under each reply."
)

if chunk_count == 0:
    st.warning(
        "Vector database is empty. Place documents in your docs folder "
        f"(`{DOCS_FOLDER_PATH}`) and click **Re-index / Sync Documents**."
    )

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 Sources & retrieved snippets", expanded=False):
                files = message.get("source_files") or []
                if files:
                    st.markdown("**Files used:** " + ", ".join(f"`{f}`" for f in files))
                for i, src in enumerate(message["sources"], start=1):
                    name = src.get("source_file", "unknown")
                    page = src.get("page")
                    header = f"**Snippet {i}** — `{name}`"
                    if page is not None:
                        header += f" (page {int(page) + 1})"
                    st.markdown(header)
                    st.text(src.get("content", ""))
                    if i < len(message["sources"]):
                        st.divider()


def _serialize_sources(docs) -> list[dict]:
    out = []
    for doc in docs:
        out.append(
            {
                "source_file": doc.metadata.get("source_file")
                or Path(str(doc.metadata.get("source", "unknown"))).name,
                "page": doc.metadata.get("page"),
                "content": doc.page_content.strip(),
            }
        )
    return out


# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & generating answer…"):
            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                result = ask_question(prompt, chat_history=history_for_llm)
            except ValueError as exc:
                # Missing API key, etc.
                result_answer = str(exc)
                result_sources: list = []
                result_files: list[str] = []
            except Exception as exc:  # noqa: BLE001
                result_answer = f"Something went wrong while answering: {exc}"
                result_sources = []
                result_files = []
            else:
                result_answer = result.answer
                result_sources = _serialize_sources(result.sources)
                result_files = result.source_files

        st.markdown(result_answer)

        if result_sources:
            with st.expander("📄 Sources & retrieved snippets", expanded=False):
                if result_files:
                    st.markdown(
                        "**Files used:** " + ", ".join(f"`{f}`" for f in result_files)
                    )
                for i, src in enumerate(result_sources, start=1):
                    name = src.get("source_file", "unknown")
                    page = src.get("page")
                    header = f"**Snippet {i}** — `{name}`"
                    if page is not None:
                        header += f" (page {int(page) + 1})"
                    st.markdown(header)
                    st.text(src.get("content", ""))
                    if i < len(result_sources):
                        st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result_answer,
            "sources": result_sources,
            "source_files": result_files,
        }
    )

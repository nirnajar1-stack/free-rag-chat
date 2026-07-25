"""
צ'אט RAG בעברית — ממשק Streamlit

ממשק בסגנון ChatGPT מעל אינדקס ChromaDB מקומי, מודל Groq,
והטמעות HuggingFace. מסמכים נטענים מ-DOCS_FOLDER_PATH או בהעלאה מהממשק.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(
    page_title="צ'אט מסמכים | RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _bootstrap_streamlit_secrets() -> None:
    """Copy Streamlit Cloud secrets into env vars before the rest of the app loads."""
    try:
        secrets = st.secrets
    except Exception:
        return

    for key in (
        "GROQ_API_KEY",
        "DOCS_FOLDER_PATH",
        "GROQ_MODEL_NAME",
        "EMBEDDING_MODEL_NAME",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVER_K",
        "RETRIEVER_MIN_SCORE",
        "RETRIEVER_MAX_SOURCES",
        "RETRIEVER_FETCH_K",
    ):
        try:
            value = secrets.get(key) if hasattr(secrets, "get") else secrets[key]
        except Exception:
            continue
        if value is not None and str(value).strip() and not os.getenv(key):
            os.environ[key] = str(value).strip()


_bootstrap_streamlit_secrets()

from src.config import DOCS_FOLDER_PATH, EMBEDDING_MODEL_NAME, GROQ_MODEL_NAME, VECTORSTORE_PATH
from src.document_loader import count_source_files
from src.indexer import reindex_documents
from src.rag_chain import ask_question
from src.uploads import save_uploaded_files
from src.vectorstore import get_chunk_count

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
    .stChatMessage, .stChatInput, .stTextInput, .stButton {
        direction: rtl;
        text-align: right;
    }
    .block-container { padding-top: 1.5rem; max-width: 900px; }
    [data-testid="stSidebar"] { min-width: 320px; }
    code, pre, [data-testid="stCode"] {
        direction: ltr;
        text-align: left;
    }
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
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


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_index_message" not in st.session_state:
        st.session_state.last_index_message = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


_init_session()


def _run_reindex() -> None:
    status = st.empty()
    progress = st.progress(0, text="מתחיל סנכרון...")
    steps = {"n": 0}

    def on_progress(msg: str) -> None:
        steps["n"] += 1
        pct = min(95, 10 + steps["n"] * 8)
        progress.progress(pct, text=msg)
        status.info(msg)

    with st.spinner("בונה אינדקס למסמכים..."):
        result = reindex_documents(progress_callback=on_progress)

    if result.success:
        progress.progress(100, text="הושלם")
        status.success(result.message)
        st.session_state.last_index_message = result.message
        st.cache_resource.clear()
        st.rerun()
    else:
        progress.progress(100, text="נכשל")
        status.error(result.message)
        st.session_state.last_index_message = result.message


with st.sidebar:
    st.title("📚 צ'אט מסמכים")
    st.caption("LangChain · Groq · HuggingFace · ChromaDB")

    st.divider()
    st.subheader("העלאת מסמכים")
    st.caption("PDF, TXT או Markdown — נשמרים בתיקיית המסמכים ואז נכנסים לאינדקס.")

    uploaded = st.file_uploader(
        "בחר/י קבצים",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help="אפשר להעלות כמה קבצים יחד",
    )

    if st.button(
        "💾 שמור וסנכרן לאינדקס",
        type="primary",
        use_container_width=True,
        disabled=not uploaded,
    ):
        saved, errors = save_uploaded_files(uploaded)
        if saved:
            st.success("נשמרו: " + ", ".join(saved))
        for err in errors:
            st.error(err)
        if saved:
            st.session_state.uploader_key += 1
            _run_reindex()

    st.divider()
    st.subheader("מסמכים")
    st.code(str(DOCS_FOLDER_PATH), language=None)

    file_counts = count_source_files()
    st.markdown(
        f"**קבצי מקור:** {file_counts['total']}  \n"
        f"PDF: {file_counts.get('.pdf', 0)} · "
        f"TXT: {file_counts.get('.txt', 0)} · "
        f"MD: {file_counts.get('.md', 0) + file_counts.get('.markdown', 0)}"
    )

    st.divider()
    st.subheader("מאגר וקטורים")
    chunk_count = get_chunk_count()
    pill_class = "ready" if chunk_count > 0 else "empty"
    label = (
        f"{chunk_count} קטעים באינדקס"
        if chunk_count > 0
        else "ריק — נדרש סנכרון"
    )
    st.markdown(
        f'<span class="status-pill {pill_class}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"נתיב שמירה: `{VECTORSTORE_PATH}`")

    st.divider()
    st.subheader("סנכרון")
    st.caption(
        "טוען את כל קבצי PDF / TXT / Markdown מתיקיית המסמכים ובונה מחדש את האינדקס."
    )

    if st.button("🔄 סנכרון / בניית אינדקס", use_container_width=True):
        _run_reindex()

    if st.session_state.last_index_message:
        st.caption(f"סנכרון אחרון: {st.session_state.last_index_message}")

    st.divider()
    st.subheader("מודלים")
    st.markdown(
        f"- **מודל שפה:** `{GROQ_MODEL_NAME}` (Groq)\n"
        f"- **הטמעות:** `{EMBEDDING_MODEL_NAME}` (מקומי)"
    )

    if st.button("🗑️ נקה שיחה", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("שאל/י את המסמכים שלך")
st.markdown(
    "התשובות מבוססות רק על הקבצים שבאינדקס. "
    "מתחת לכל תשובה יופיעו המקורות הרלוונטיים בלבד."
)

if chunk_count == 0:
    st.warning(
        "מאגר הווקטורים ריק. העלה/י מסמכים בסרגל הצד "
        f"או הוסף/י קבצים ל-`{DOCS_FOLDER_PATH}`, ואז לחץ/י על "
        "**סנכרון / בניית אינדקס**."
    )


def _preview_text(text: str, limit: int = 260) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def _render_sources(sources: list[dict], source_files: list[str] | None = None) -> None:
    if not sources:
        return

    grouped: dict[str, list[dict]] = {}
    for src in sources:
        name = src.get("source_file", "לא ידוע")
        grouped.setdefault(name, []).append(src)

    files = source_files or list(grouped.keys())
    with st.expander(f"📄 מקורות רלוונטיים ({len(files)})", expanded=False):
        for file_name in files:
            chunks = grouped.get(file_name) or []
            if not chunks:
                continue
            st.markdown(f"**{file_name}**")
            for i, src in enumerate(chunks, start=1):
                page = src.get("page")
                relevance = src.get("relevance")
                meta_parts: list[str] = []
                if page is not None:
                    meta_parts.append(f"עמוד {int(page) + 1}")
                if relevance is not None:
                    meta_parts.append(f"התאמה {float(relevance):.0%}")
                caption = f"קטע {i}"
                if meta_parts:
                    caption += " · " + " · ".join(meta_parts)
                st.caption(caption)
                st.markdown(f"> {_preview_text(src.get('content', ''))}")
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            _render_sources(message["sources"], message.get("source_files"))


def _serialize_sources(docs) -> list[dict]:
    out = []
    for doc in docs:
        out.append(
            {
                "source_file": doc.metadata.get("source_file")
                or Path(str(doc.metadata.get("source", "לא ידוע"))).name,
                "page": doc.metadata.get("page"),
                "relevance": doc.metadata.get("relevance"),
                "content": doc.page_content.strip(),
            }
        )
    return out


if prompt := st.chat_input("כתוב/י שאלה על המסמכים..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("שולף הקשר ומייצר תשובה..."):
            history_for_llm = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                result = ask_question(prompt, chat_history=history_for_llm)
            except ValueError as exc:
                result_answer = str(exc)
                result_sources: list = []
                result_files: list[str] = []
            except Exception as exc:  # noqa: BLE001
                result_answer = f"אירעה שגיאה בעת יצירת התשובה: {exc}"
                result_sources = []
                result_files = []
            else:
                result_answer = result.answer
                result_sources = _serialize_sources(result.sources)
                result_files = result.source_files

        st.markdown(result_answer)
        _render_sources(result_sources, result_files)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result_answer,
            "sources": result_sources,
            "source_files": result_files,
        }
    )

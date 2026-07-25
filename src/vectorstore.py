"""ChromaDB persistent vector store helpers (Windows-safe reset)."""

from __future__ import annotations

import gc
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import RETRIEVER_K, VECTORSTORE_PATH
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_documents"

# Single live client — avoids opening many SQLite locks under Streamlit.
_store: Chroma | None = None
_store_path: str | None = None


def _release_store() -> None:
    """Drop the cached Chroma client so Windows can unlock SQLite files."""
    global _store, _store_path
    _store = None
    _store_path = None
    gc.collect()


def get_vectorstore(persist_directory: Path | None = None) -> Chroma:
    """Open (or reuse) the persistent Chroma vector store."""
    global _store, _store_path

    path = Path(persist_directory) if persist_directory else VECTORSTORE_PATH
    path.mkdir(parents=True, exist_ok=True)
    path_str = str(path.resolve())

    if _store is not None and _store_path == path_str:
        return _store

    if _store is not None:
        _release_store()

    _store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=path_str,
    )
    _store_path = path_str
    return _store


def get_retriever(
    k: int = RETRIEVER_K,
    persist_directory: Path | None = None,
) -> VectorStoreRetriever:
    """Return a similarity-search retriever over the persistent store."""
    store = get_vectorstore(persist_directory)
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def retrieve_relevant_documents(
    query: str,
    *,
    fetch_k: int = 8,
    min_score: float = 0.28,
    max_docs: int = 3,
    persist_directory: Path | None = None,
) -> list[Document]:
    """
    Retrieve chunks with relevance filtering.

    Uses relevance scores (higher = better), drops weak matches, keeps only
    chunks close to the best score, and caps how many are returned.
    """
    store = get_vectorstore(persist_directory)
    try:
        scored = store.similarity_search_with_relevance_scores(query, k=fetch_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("relevance search failed (%s); falling back to plain search", exc)
        docs = store.similarity_search(query, k=max_docs)
        for doc in docs:
            doc.metadata["relevance"] = None
        return docs

    usable: list[tuple[Document, float]] = []
    for doc, score in scored:
        if score is None:
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        # Some backends invert scores; clamp into a usable range.
        if score_f < 0:
            continue
        usable.append((doc, score_f))

    if not usable:
        return []

    usable.sort(key=lambda item: item[1], reverse=True)
    best = usable[0][1]

    # Keep only reasonably strong matches near the top hit.
    filtered = [
        (doc, score)
        for doc, score in usable
        if score >= min_score and score >= (best - 0.18)
    ]
    if not filtered:
        filtered = [usable[0]]

    selected: list[Document] = []
    for doc, score in filtered[:max_docs]:
        doc.metadata["relevance"] = round(score, 3)
        selected.append(doc)
    return selected


def get_chunk_count(persist_directory: Path | None = None) -> int:
    """Return the number of indexed chunks, or 0 if the store is empty/missing."""
    path = Path(persist_directory) if persist_directory else VECTORSTORE_PATH
    if not path.exists():
        return 0
    try:
        if not any(path.iterdir()):
            return 0
    except OSError:
        return 0

    try:
        store = get_vectorstore(path)
        return int(store._collection.count())  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read chunk count: %s", exc)
        return 0


def _clear_collection_contents(store: Chroma) -> None:
    """Remove every document without deleting chroma.sqlite3 on disk."""
    try:
        data = store.get()
        ids = list(data.get("ids") or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("store.get() failed while clearing: %s", exc)
        ids = []

    if ids:
        batch_size = 200
        for start in range(0, len(ids), batch_size):
            store.delete(ids=ids[start : start + batch_size])
        return

    # Empty or unreadable — recreate collection handle
    try:
        store.delete_collection()
    except Exception as exc:  # noqa: BLE001
        logger.warning("delete_collection failed: %s", exc)
    _release_store()


def clear_vectorstore(persist_directory: Path | None = None) -> None:
    """
    Clear indexed data in a Windows-safe way.

    Prefer deleting collection contents through the open Chroma client.
    Folder deletion is only a last-resort fallback after releasing locks.
    """
    path = Path(persist_directory) if persist_directory else VECTORSTORE_PATH
    path.mkdir(parents=True, exist_ok=True)

    try:
        store = get_vectorstore(path)
        _clear_collection_contents(store)
        logger.info("Cleared Chroma collection contents at %s", path)
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("API clear failed, falling back to folder delete: %s", exc)

    _release_store()
    time.sleep(0.3)

    if path.exists():
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                shutil.rmtree(path)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.4 * (attempt + 1))
                _release_store()
        if last_error is not None:
            raise RuntimeError(
                "לא ניתן לנקות את מאגר הווקטורים כי הקובץ תפוס "
                "(כנראה על ידי חלון Streamlit נוסף). סגור/י כפילויות ונסה/י שוב. "
                f"פרטים: {last_error}"
            ) from last_error

    path.mkdir(parents=True, exist_ok=True)
    logger.info("Cleared vector store directory at %s", path)


def index_documents(
    chunks: list[Document],
    persist_directory: Path | None = None,
    *,
    clear_existing: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> Chroma:
    """
    Embed and persist document chunks into ChromaDB.

    Uses in-place collection clear (not folder delete) so Windows + Streamlit
    do not hit WinError 32 on chroma.sqlite3.
    """
    path = Path(persist_directory) if persist_directory else VECTORSTORE_PATH
    path.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    if not chunks:
        raise ValueError(
            "אין קטעים לאינדקס. הוסף/י קבצי PDF, TXT או Markdown "
            "לתיקיית המסמכים ונסה/י שוב."
        )

    if clear_existing:
        _progress("מנקה את מאגר הווקטורים הקיים...")
        clear_vectorstore(path)

    store = get_vectorstore(path)
    _progress(f"מטמיע ומאנדקס {len(chunks)} קטעים...")

    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        end = min(start + batch_size, len(chunks))
        _progress(f"מאנדקס קטעים {start + 1}-{end} מתוך {len(chunks)}...")
        store.add_documents(batch)

    _progress(f"האינדוקס הושלם - {get_chunk_count(path)} קטעים נשמרו.")
    return store

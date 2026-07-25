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

from src.config import RETRIEVER_K
from src.embeddings import get_embeddings
from src.storage import get_vectorstore_path

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

    path = Path(persist_directory) if persist_directory else get_vectorstore_path()
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
    min_score: float = 0.0,
    max_docs: int = 4,
    persist_directory: Path | None = None,
) -> list[Document]:
    """
    Retrieve the strongest matching chunks.

    Uses Chroma distances (lower = better). Avoids LangChain "relevance scores"
    which can be negative/invalid with this embedding setup and caused empty
    retrieval for Hebrew queries.
    """
    del min_score  # kept for API compatibility; distance ranking is primary
    store = get_vectorstore(persist_directory)

    try:
        scored = store.similarity_search_with_score(query, k=fetch_k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scored search failed (%s); falling back to plain search", exc)
        docs = store.similarity_search(query, k=max_docs)
        for doc in docs:
            doc.metadata["relevance"] = None
        return docs

    if not scored:
        return []

    # Distance: lower is better
    scored.sort(key=lambda item: float(item[1]))
    best_distance = float(scored[0][1])

    # Keep neighbors close to the best hit (absolute margin on distance).
    margin = 0.45
    filtered = [
        (doc, float(dist))
        for doc, dist in scored
        if float(dist) <= best_distance + margin
    ]
    if not filtered:
        filtered = [scored[0]]

    selected: list[Document] = []
    for doc, dist in filtered[:max_docs]:
        # Convert distance to a rough 0-1 display score (not used for filtering).
        approx = max(0.0, min(1.0, 1.0 / (1.0 + dist)))
        doc.metadata["relevance"] = round(approx, 3)
        doc.metadata["distance"] = round(dist, 4)
        selected.append(doc)
    return selected


def get_chunk_count(persist_directory: Path | None = None) -> int:
    """Return the number of indexed chunks, or 0 if the store is empty/missing."""
    path = Path(persist_directory) if persist_directory else get_vectorstore_path()
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
    path = Path(persist_directory) if persist_directory else get_vectorstore_path()
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
    path = Path(persist_directory) if persist_directory else get_vectorstore_path()
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

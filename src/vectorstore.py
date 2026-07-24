"""ChromaDB persistent vector store helpers."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import RETRIEVER_K, VECTORSTORE_PATH
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_documents"


def get_vectorstore(persist_directory: Path | None = None) -> Chroma:
    """Open (or create) the persistent Chroma vector store."""
    path = Path(persist_directory) if persist_directory else VECTORSTORE_PATH
    path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(path),
    )


def get_retriever(
    k: int = RETRIEVER_K,
    persist_directory: Path | None = None,
) -> VectorStoreRetriever:
    """Return a similarity-search retriever over the persistent store."""
    store = get_vectorstore(persist_directory)
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_chunk_count(persist_directory: Path | None = None) -> int:
    """Return the number of indexed chunks, or 0 if the store is empty/missing."""
    path = Path(persist_directory) if persist_directory else VECTORSTORE_PATH
    if not path.exists() or not any(path.iterdir()):
        return 0

    try:
        store = get_vectorstore(path)
        return int(store._collection.count())  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read chunk count: %s", exc)
        return 0


def clear_vectorstore(persist_directory: Path | None = None) -> None:
    """Delete the persistent vector store directory (full rebuild)."""
    path = Path(persist_directory) if persist_directory else VECTORSTORE_PATH
    if path.exists():
        shutil.rmtree(path)
        logger.info("Cleared vector store at %s", path)
    path.mkdir(parents=True, exist_ok=True)


def index_documents(
    chunks: list[Document],
    persist_directory: Path | None = None,
    *,
    clear_existing: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> Chroma:
    """
    Embed and persist document chunks into ChromaDB.

    Args:
        chunks: Pre-split LangChain documents.
        persist_directory: Override for the default ``data/vectorstore/``.
        clear_existing: If True, wipe the store before indexing (clean sync).
        progress_callback: Optional ``callable(message: str)`` for UI updates.

    Returns:
        The populated Chroma vector store.
    """
    path = Path(persist_directory) if persist_directory else VECTORSTORE_PATH

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    if not chunks:
        raise ValueError(
            "No document chunks to index. Add PDF, TXT, or Markdown files "
            "to your DOCS_FOLDER_PATH and try again."
        )

    if clear_existing:
        _progress("Clearing existing vector store...")
        clear_vectorstore(path)

    _progress(f"Embedding and indexing {len(chunks)} chunk(s)...")
    embeddings = get_embeddings()

    batch_size = 64
    store: Chroma | None = None

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        end = min(start + batch_size, len(chunks))
        _progress(f"Indexing chunks {start + 1}-{end} of {len(chunks)}...")

        if store is None:
            store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(path),
            )
        else:
            store.add_documents(batch)

    assert store is not None
    _progress(f"Indexing complete - {get_chunk_count(path)} chunk(s) stored.")
    return store

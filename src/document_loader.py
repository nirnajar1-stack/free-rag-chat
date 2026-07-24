"""Load PDF, TXT, and Markdown documents from a configurable folder."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_FOLDER_PATH,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


def _source_name(doc: Document) -> str:
    """Extract a clean filename from document metadata."""
    source = doc.metadata.get("source", "unknown")
    return Path(source).name


def _load_with_loader(
    folder: Path,
    glob: str,
    loader_cls: type,
    loader_kwargs: dict | None = None,
) -> list[Document]:
    """Load documents matching ``glob`` under ``folder`` using ``loader_cls``."""
    if not folder.exists():
        return []

    kwargs = loader_kwargs or {}
    try:
        loader = DirectoryLoader(
            str(folder),
            glob=glob,
            loader_cls=loader_cls,
            loader_kwargs=kwargs,
            recursive=True,
            show_progress=False,
            use_multithreading=False,
            silent_errors=True,
        )
        docs = loader.load()
        logger.info("Loaded %d document(s) for pattern %s", len(docs), glob)
        return docs
    except Exception as exc:  # noqa: BLE001 — surface per-format failures
        logger.warning("Failed loading %s from %s: %s", glob, folder, exc)
        return []


def load_documents(folder: Path | None = None) -> list[Document]:
    """
    Load all supported documents from ``folder`` (or ``DOCS_FOLDER_PATH``).

    Supports PDF, TXT, and Markdown (``.md`` / ``.markdown``).
    """
    docs_path = Path(folder) if folder is not None else DOCS_FOLDER_PATH

    if not docs_path.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {docs_path}\n"
            "Set DOCS_FOLDER_PATH in .env to your Google Drive sync folder "
            "or create data/docs/ and add PDF/TXT/MD files."
        )

    if not docs_path.is_dir():
        raise NotADirectoryError(f"DOCS_FOLDER_PATH is not a directory: {docs_path}")

    documents: list[Document] = []

    documents.extend(_load_with_loader(docs_path, "**/*.pdf", PyPDFLoader))

    text_kwargs = {"encoding": "utf-8", "autodetect_encoding": True}
    documents.extend(
        _load_with_loader(docs_path, "**/*.txt", TextLoader, text_kwargs)
    )
    documents.extend(
        _load_with_loader(docs_path, "**/*.md", TextLoader, text_kwargs)
    )
    documents.extend(
        _load_with_loader(docs_path, "**/*.markdown", TextLoader, text_kwargs)
    )

    for doc in documents:
        doc.metadata["source_file"] = _source_name(doc)
        ext = Path(doc.metadata.get("source", "")).suffix.lower()
        doc.metadata["file_type"] = ext.lstrip(".") if ext else "unknown"

    documents = [d for d in documents if d.page_content and d.page_content.strip()]

    logger.info(
        "Total documents loaded from %s: %d (extensions: %s)",
        docs_path,
        len(documents),
        ", ".join(SUPPORTED_EXTENSIONS),
    )
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata.setdefault("source_file", _source_name(chunk))
        chunk.metadata["chunk_id"] = i

    logger.info(
        "Split %d document(s) into %d chunk(s) (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def count_source_files(folder: Path | None = None) -> dict[str, int]:
    """Count supported files in the docs folder by extension."""
    docs_path = Path(folder) if folder is not None else DOCS_FOLDER_PATH
    counts: dict[str, int] = {ext: 0 for ext in SUPPORTED_EXTENSIONS}
    counts["total"] = 0

    if not docs_path.exists():
        return counts

    for path in docs_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            ext = path.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1
            counts["total"] += 1

    return counts

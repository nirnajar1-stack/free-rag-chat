"""סנכרון מחדש של מסמכים מ-DOCS_FOLDER_PATH ל-ChromaDB."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.config import DOCS_FOLDER_PATH
from src.document_loader import count_source_files, load_documents, split_documents
from src.vectorstore import get_chunk_count, index_documents

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """סיכום הרצת סנכרון / בניית אינדקס."""

    docs_folder: Path
    source_file_count: int
    document_count: int
    chunk_count: int
    success: bool
    message: str


def reindex_documents(
    folder: Path | None = None,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> IndexResult:
    """סנכרון מלא: טעינה -> פיצול -> ניקוי Chroma -> הטמעה -> שמירה."""
    docs_path = Path(folder) if folder is not None else DOCS_FOLDER_PATH

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        file_counts = count_source_files(docs_path)
        _progress(
            f"סורק את {docs_path} - נמצאו {file_counts['total']} קבצים נתמכים..."
        )

        _progress("טוען מסמכים (PDF, TXT, Markdown)...")
        documents = load_documents(docs_path)

        if not documents:
            return IndexResult(
                docs_folder=docs_path,
                source_file_count=file_counts["total"],
                document_count=0,
                chunk_count=get_chunk_count(),
                success=False,
                message=(
                    f"לא נמצא תוכן קריא ב-{docs_path}. "
                    "הוסף/י קבצי .pdf, .txt או .md ונסה/י שוב."
                ),
            )

        _progress(f"מפצל {len(documents)} עמודים לקטעים...")
        chunks = split_documents(documents)

        index_documents(
            chunks,
            clear_existing=True,
            progress_callback=progress_callback,
        )

        final_count = get_chunk_count()
        message = (
            f"סונכרנו {file_counts['total']} קבצים -> "
            f"{len(documents)} עמודים -> {final_count} קטעים באינדקס."
        )
        _progress(message)

        return IndexResult(
            docs_folder=docs_path,
            source_file_count=file_counts["total"],
            document_count=len(documents),
            chunk_count=final_count,
            success=True,
            message=message,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Re-index failed")
        return IndexResult(
            docs_folder=docs_path,
            source_file_count=0,
            document_count=0,
            chunk_count=get_chunk_count(),
            success=False,
            message=f"הסנכרון נכשל: {exc}",
        )

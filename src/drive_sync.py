"""סנכרון Google Drive + בניית אינדקס אוטומטית."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.google_drive import (
    get_drive_folder_id,
    is_drive_configured,
    sync_drive_folder_to_local,
    upload_bytes_to_drive,
)
from src.indexer import IndexResult, reindex_documents
from src.storage import ensure_storage_dirs

logger = logging.getLogger(__name__)


@dataclass
class DriveSyncResult:
    configured: bool
    synced: bool
    reindexed: bool
    message: str
    downloaded: list[str]
    index_result: IndexResult | None = None


def sync_drive_and_reindex_if_needed(
    *,
    force_reindex: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> DriveSyncResult:
    """
    Pull latest files from the cloud Drive folder and reindex when something changed.
    """
    if not is_drive_configured():
        return DriveSyncResult(
            configured=False,
            synced=False,
            reindexed=False,
            message="Google Drive בענן לא מוגדר עדיין (חסרים Folder ID / Service Account).",
            downloaded=[],
        )

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    ensure_storage_dirs()
    _progress("מסנכרן קבצים מ-Google Drive בענן...")
    summary = sync_drive_folder_to_local()

    downloaded = summary.get("downloaded") or []
    need = bool(summary.get("need_reindex")) or force_reindex

    if not need:
        return DriveSyncResult(
            configured=True,
            synced=True,
            reindexed=False,
            message=(
                f"סנכרון Drive הושלם — אין שינויים "
                f"({summary.get('remote_count', 0)} קבצים בענן)."
            ),
            downloaded=[],
        )

    _progress(
        f"זוהו שינויים ({len(downloaded)} קבצים חדשים/מעודכנים) — בונה אינדקס..."
    )
    index_result = reindex_documents(progress_callback=progress_callback)
    return DriveSyncResult(
        configured=True,
        synced=True,
        reindexed=index_result.success,
        message=(
            f"Drive: הורדו {len(downloaded)} קבצים. {index_result.message}"
            if downloaded
            else index_result.message
        ),
        downloaded=downloaded,
        index_result=index_result,
    )


def upload_to_drive_and_reindex(
    filename: str,
    data: bytes,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> DriveSyncResult:
    """Upload one file to Drive cloud, sync locally, and rebuild the index."""
    if not is_drive_configured():
        return DriveSyncResult(
            configured=False,
            synced=False,
            reindexed=False,
            message="Google Drive בענן לא מוגדר — לא ניתן להעלות ישירות לדרייב.",
            downloaded=[],
        )

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    _progress(f"מעלה ל-Google Drive: {filename}")
    remote = upload_bytes_to_drive(filename, data)
    _progress(f"הועלה לדרייב: {remote.name} — מסנכרן ובונה אינדקס...")
    return sync_drive_and_reindex_if_needed(
        force_reindex=True,
        progress_callback=progress_callback,
    )


def upload_many_to_drive_and_reindex(
    files: list[tuple[str, bytes]],
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> DriveSyncResult:
    """Upload multiple files to Drive, then sync + reindex once."""
    if not is_drive_configured():
        return DriveSyncResult(
            configured=False,
            synced=False,
            reindexed=False,
            message="Google Drive בענן לא מוגדר.",
            downloaded=[],
        )

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    names: list[str] = []
    for name, data in files:
        _progress(f"מעלה ל-Drive: {name}")
        remote = upload_bytes_to_drive(name, data)
        names.append(remote.name)

    _progress("כל הקבצים הועלו — מסנכרן ובונה אינדקס...")
    result = sync_drive_and_reindex_if_needed(
        force_reindex=True,
        progress_callback=progress_callback,
    )
    result.downloaded = names
    result.message = f"הועלו ל-Drive: {', '.join(names)}. {result.message}"
    return result

"""Google Drive cloud sync via Service Account (no desktop install)."""

from __future__ import annotations

import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, SUPPORTED_EXTENSIONS, _secret
from src.storage import get_docs_folder

logger = logging.getLogger(__name__)

SYNC_STATE_PATH = PROJECT_ROOT / "data" / "drive_sync_state.json"
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


@dataclass
class DriveFile:
    id: str
    name: str
    modified_time: str
    mime_type: str
    size: str | None = None


def _credentials_info() -> dict[str, Any] | None:
    """Load service-account JSON from env, secrets, or a local file path."""
    # 1) Full JSON string in env
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw and raw.strip().startswith("{"):
        return json.loads(raw)

    # 2) Path to JSON file
    path = (
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or _secret("GOOGLE_APPLICATION_CREDENTIALS")
        or raw
    )
    if path and Path(str(path)).expanduser().is_file():
        return json.loads(Path(str(path)).expanduser().read_text(encoding="utf-8"))

    # 3) Streamlit secrets — string or mapped object
    try:
        import streamlit as st

        secrets = st.secrets
        if "GOOGLE_SERVICE_ACCOUNT_JSON" in secrets:
            val = secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
            if isinstance(val, str):
                return json.loads(val)
            # AttrDict / dict-like from TOML table
            return dict(val)
    except Exception:  # noqa: BLE001
        pass

    return None


def get_drive_folder_id() -> str | None:
    value = _secret("GOOGLE_DRIVE_FOLDER_ID")
    if value and str(value).strip():
        return str(value).strip()
    # Also allow saved setting
    from src.storage import _load_settings_file

    saved = _load_settings_file().get("google_drive_folder_id")
    if saved and str(saved).strip():
        return str(saved).strip()
    return None


def save_drive_folder_id(folder_id: str) -> None:
    from src.storage import SETTINGS_PATH, _load_settings_file

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _load_settings_file()
    data["google_drive_folder_id"] = folder_id.strip()
    SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.environ["GOOGLE_DRIVE_FOLDER_ID"] = folder_id.strip()


def is_drive_configured() -> bool:
    return bool(_credentials_info() and get_drive_folder_id())


def get_service_account_email() -> str | None:
    info = _credentials_info()
    if not info:
        return None
    return info.get("client_email")


def _build_service():
    info = _credentials_info()
    if not info:
        raise ValueError(
            "חסרים פרטי Service Account. הוסף/י GOOGLE_SERVICE_ACCOUNT_JSON "
            "ב-.env או ב-Secrets של Streamlit."
        )

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _sanitize_name(name: str) -> str:
    base = Path(name).name.strip() or "document"
    base = _UNSAFE.sub("_", base).lstrip(".")
    return base or "document"


def list_drive_documents(folder_id: str | None = None) -> list[DriveFile]:
    """List supported files in the shared Drive folder."""
    folder_id = folder_id or get_drive_folder_id()
    if not folder_id:
        raise ValueError("חסר GOOGLE_DRIVE_FOLDER_ID")

    service = _build_service()
    query = (
        f"'{folder_id}' in parents and trashed = false "
        "and (mimeType = 'application/pdf' "
        "or mimeType = 'text/plain' "
        "or mimeType = 'text/markdown' "
        "or mimeType contains 'text/' "
        "or name contains '.pdf' "
        "or name contains '.txt' "
        "or name contains '.md')"
    )

    files: list[DriveFile] = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                pageToken=page_token,
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for item in response.get("files", []):
            name = item.get("name") or ""
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                # Allow Google Docs export? skip for now — only binary/text uploads
                continue
            files.append(
                DriveFile(
                    id=item["id"],
                    name=name,
                    modified_time=item.get("modifiedTime") or "",
                    mime_type=item.get("mimeType") or "",
                    size=item.get("size"),
                )
            )
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files


def upload_bytes_to_drive(
    filename: str,
    data: bytes,
    folder_id: str | None = None,
) -> DriveFile:
    """Upload a file into the shared Drive folder (creates or updates by name)."""
    from googleapiclient.http import MediaIoBaseUpload

    folder_id = folder_id or get_drive_folder_id()
    if not folder_id:
        raise ValueError("חסר GOOGLE_DRIVE_FOLDER_ID")

    safe_name = _sanitize_name(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"סוג קובץ לא נתמך: {safe_name}")

    service = _build_service()
    mime = MIME_BY_EXT.get(suffix, "application/octet-stream")

    # If a file with same name exists — update it
    existing = [
        f
        for f in list_drive_documents(folder_id)
        if f.name == safe_name
    ]
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)

    if existing:
        updated = (
            service.files()
            .update(
                fileId=existing[0].id,
                media_body=media,
                fields="id, name, mimeType, modifiedTime, size",
                supportsAllDrives=True,
            )
            .execute()
        )
        item = updated
    else:
        metadata = {"name": safe_name, "parents": [folder_id]}
        created = (
            service.files()
            .create(
                body=metadata,
                media_body=media,
                fields="id, name, mimeType, modifiedTime, size",
                supportsAllDrives=True,
            )
            .execute()
        )
        item = created

    return DriveFile(
        id=item["id"],
        name=item.get("name") or safe_name,
        modified_time=item.get("modifiedTime") or "",
        mime_type=item.get("mimeType") or mime,
        size=item.get("size"),
    )


def _load_sync_state() -> dict[str, Any]:
    if not SYNC_STATE_PATH.exists():
        return {"files": {}}
    try:
        return json.loads(SYNC_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"files": {}}


def _save_sync_state(state: dict[str, Any]) -> None:
    SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sync_drive_folder_to_local(
    folder_id: str | None = None,
    *,
    local_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Download new/changed Drive files into the local docs folder.

    Returns a summary: downloaded, unchanged, removed_locally, need_reindex.
    """
    from googleapiclient.http import MediaIoBaseDownload

    folder_id = folder_id or get_drive_folder_id()
    dest = Path(local_dir) if local_dir else get_docs_folder()
    dest.mkdir(parents=True, exist_ok=True)

    remote_files = list_drive_documents(folder_id)
    state = _load_sync_state()
    known: dict[str, Any] = state.get("files") or {}

    service = _build_service()
    downloaded: list[str] = []
    unchanged: list[str] = []
    remote_ids = set()

    for remote in remote_files:
        remote_ids.add(remote.id)
        safe_name = _sanitize_name(remote.name)
        target = dest / safe_name
        prev = known.get(remote.id) or {}

        if (
            prev.get("modified_time") == remote.modified_time
            and prev.get("local_name") == safe_name
            and target.exists()
        ):
            unchanged.append(safe_name)
            continue

        request = service.files().get_media(fileId=remote.id, supportsAllDrives=True)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        target.write_bytes(buffer.getvalue())
        known[remote.id] = {
            "modified_time": remote.modified_time,
            "local_name": safe_name,
            "name": remote.name,
        }
        downloaded.append(safe_name)

    # Drop state entries for files removed from Drive (keep local copies)
    stale = [fid for fid in list(known) if fid not in remote_ids]
    for fid in stale:
        known.pop(fid, None)

    state["files"] = known
    state["folder_id"] = folder_id
    _save_sync_state(state)

    need_reindex = bool(downloaded) or bool(stale)
    return {
        "downloaded": downloaded,
        "unchanged": unchanged,
        "removed_from_drive": len(stale),
        "need_reindex": need_reindex,
        "remote_count": len(remote_files),
        "local_dir": str(dest),
    }

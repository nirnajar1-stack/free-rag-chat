"""אחסון מסמכים ואינדקס — כולל תיקיית Google Drive for desktop."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import PROJECT_ROOT, _secret

logger = logging.getLogger(__name__)

SETTINGS_PATH = PROJECT_ROOT / "data" / "app_settings.json"
DEFAULT_DOCS = PROJECT_ROOT / "data" / "docs"
DEFAULT_VECTORSTORE = PROJECT_ROOT / "data" / "vectorstore"
DRIVE_VECTORSTORE_DIRNAME = ".rag_vectorstore"


def _load_settings_file() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read settings file: %s", exc)
        return {}


def save_docs_folder(path: str | Path) -> Path:
    """Persist the chosen docs folder (e.g. Google Drive sync path)."""
    resolved = Path(str(path).strip().strip('"')).expanduser()
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _load_settings_file()
    data["docs_folder_path"] = str(resolved)
    SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Keep process env in sync for modules that read os.getenv
    import os

    os.environ["DOCS_FOLDER_PATH"] = str(resolved)
    return resolved


def get_docs_folder() -> Path:
    """
    Resolve the active documents folder.

    Priority: saved UI setting → env/secrets → local data/docs
    """
    saved = _load_settings_file().get("docs_folder_path")
    if saved and str(saved).strip():
        return Path(str(saved)).expanduser()

    from_env = _secret("DOCS_FOLDER_PATH")
    if from_env and str(from_env).strip():
        return Path(str(from_env)).expanduser()

    return DEFAULT_DOCS


def get_vectorstore_path() -> Path:
    """
    Resolve Chroma persistence directory.

    If VECTORSTORE_PATH is set explicitly, use it.
    If docs live outside the project (typical Google Drive folder), store
    Chroma under ``{docs}/.rag_vectorstore`` so everything syncs with Drive.
    Otherwise use local ``data/vectorstore``.
    """
    explicit = _secret("VECTORSTORE_PATH")
    if explicit and str(explicit).strip():
        return Path(str(explicit)).expanduser()

    docs = get_docs_folder().resolve()
    try:
        docs.relative_to(PROJECT_ROOT.resolve())
        # Docs are inside the project → keep local vectorstore
        return DEFAULT_VECTORSTORE
    except ValueError:
        # Docs are outside the project (e.g. G:/My Drive/...) → Drive storage
        return docs / DRIVE_VECTORSTORE_DIRNAME


def ensure_storage_dirs() -> tuple[Path, Path]:
    """Create docs + vectorstore directories if missing."""
    docs = get_docs_folder()
    store = get_vectorstore_path()
    docs.mkdir(parents=True, exist_ok=True)
    store.mkdir(parents=True, exist_ok=True)
    return docs, store


def is_google_drive_path(path: Path | None = None) -> bool:
    """Heuristic: path looks like a Google Drive for desktop location."""
    p = str(path or get_docs_folder()).replace("\\", "/").lower()
    return (
        "my drive" in p
        or "google drive" in p
        or "cloudstorage/googledrive" in p
        or p.startswith("g:/")
        or "/g:" in p
    )


def describe_storage() -> dict[str, str | bool]:
    docs = get_docs_folder()
    store = get_vectorstore_path()
    return {
        "docs_folder": str(docs),
        "vectorstore": str(store),
        "on_google_drive": is_google_drive_path(docs),
        "docs_exists": docs.exists(),
    }

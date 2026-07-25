"""Save Streamlit-uploaded documents into the docs folder."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import SUPPORTED_EXTENSIONS
from src.storage import get_docs_folder

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_filename(name: str) -> str:
    """Return a safe basename that keeps Hebrew / unicode letters."""
    base = Path(name).name.strip() or "document"
    base = _UNSAFE.sub("_", base)
    # Prevent hidden / traversal style names
    base = base.lstrip(".")
    if not base:
        base = "document"
    return base


def save_uploaded_files(
    uploaded_files: list,
    destination: Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    Persist uploaded Streamlit files under the docs folder.

    Returns:
        (saved_names, skipped_or_error_messages)
    """
    dest = Path(destination) if destination is not None else get_docs_folder()
    dest.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    errors: list[str] = []

    for upload in uploaded_files:
        original = getattr(upload, "name", "document")
        safe_name = sanitize_filename(original)
        suffix = Path(safe_name).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            errors.append(f"סוג קובץ לא נתמך: {original}")
            continue

        target = dest / safe_name
        # Avoid overwrite collisions
        if target.exists():
            stem, ext = target.stem, target.suffix
            n = 1
            while True:
                candidate = dest / f"{stem}_{n}{ext}"
                if not candidate.exists():
                    target = candidate
                    break
                n += 1

        try:
            data = upload.getvalue()
            target.write_bytes(data)
            saved.append(target.name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"שגיאה בשמירת {original}: {exc}")

    return saved, errors

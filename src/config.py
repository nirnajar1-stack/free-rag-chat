"""Application configuration loaded from environment / Streamlit secrets."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _secret(key: str, default: str | None = None) -> str | None:
    """Read from env first, then Streamlit Cloud / local secrets.toml."""
    value = os.getenv(key)
    if value is not None and str(value).strip() != "":
        return str(value).strip()

    try:
        import streamlit as st

        secrets = st.secrets
        secret_val = None
        try:
            secret_val = secrets[key]
        except Exception:
            try:
                secret_val = secrets.get(key)  # type: ignore[attr-defined]
            except Exception:
                secret_val = None
        if secret_val is not None and str(secret_val).strip() != "":
            return str(secret_val).strip()
    except Exception:  # noqa: BLE001 — secrets unavailable outside Streamlit
        pass

    return default


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = _PROJECT_ROOT
DOCS_FOLDER_PATH: Path = Path(
    _secret("DOCS_FOLDER_PATH", str(_PROJECT_ROOT / "data" / "docs"))
).expanduser()
VECTORSTORE_PATH: Path = _PROJECT_ROOT / "data" / "vectorstore"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = _secret(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
) or "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME: str = (
    _secret("GROQ_MODEL_NAME", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile"
)
GROQ_API_KEY: str | None = _secret("GROQ_API_KEY")

# ---------------------------------------------------------------------------
# Chunking & retrieval
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(_secret("CHUNK_SIZE", "1000") or "1000")
CHUNK_OVERLAP: int = int(_secret("CHUNK_OVERLAP", "200") or "200")
RETRIEVER_K: int = int(_secret("RETRIEVER_K", "4") or "4")
# Soft floor only (applied when the best hit is already strong). Keep low for Hebrew/OCR.
RETRIEVER_MIN_SCORE: float = float(_secret("RETRIEVER_MIN_SCORE", "0.05") or "0.05")
RETRIEVER_MAX_SOURCES: int = int(_secret("RETRIEVER_MAX_SOURCES", "4") or "4")
RETRIEVER_FETCH_K: int = int(_secret("RETRIEVER_FETCH_K", "8") or "8")

# ---------------------------------------------------------------------------
# Supported document extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt", ".md", ".markdown")


def validate_groq_api_key() -> str:
    """Return the Groq API key or raise a clear error if missing."""
    # Re-read at call time so Streamlit secrets are picked up after app boot
    key = _secret("GROQ_API_KEY") or GROQ_API_KEY
    if not key or not key.strip():
        raise ValueError(
            "חסר מפתח GROQ_API_KEY. מקומית: העתק/י .env.example ל-.env. "
            "ב-Streamlit Cloud / HF Spaces: הוסף/י GROQ_API_KEY בסודות האפליקציה. "
            "מפתח חינמי: https://console.groq.com/keys"
        )
    return key.strip()

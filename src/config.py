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
        return str(value)

    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            secret_val = st.secrets[key]
            if secret_val is not None and str(secret_val).strip() != "":
                return str(secret_val)
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
            "GROQ_API_KEY is not set. For local use: copy .env.example to .env. "
            "For Streamlit Cloud / HF Spaces: add GROQ_API_KEY in app secrets. "
            "Get a free key at https://console.groq.com/keys"
        )
    return key.strip()

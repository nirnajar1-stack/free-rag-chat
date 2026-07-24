"""Local HuggingFace embeddings (free, no API key required)."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFace embeddings instance.

    Uses ``all-MiniLM-L6-v2`` by default — runs entirely on CPU locally,
    so indexing and retrieval stay free.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

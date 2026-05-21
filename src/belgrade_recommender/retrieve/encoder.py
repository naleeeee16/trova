"""Lazy wrapper around sentence-transformers (optional dependency)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


class SentenceEncoder:
    """Small sentence embedding model; first encode() call loads weights."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = (
            model_name or os.environ.get("SENTENCE_TRANSFORMER_MODEL") or "paraphrase-multilingual-MiniLM-L12-v2"
        ).strip()
        self._model: Any = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                'Install retrieval extras: pip install -e ".[retrieve]"',
            ) from exc
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

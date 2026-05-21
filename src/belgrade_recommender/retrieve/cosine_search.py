"""Top-k cosine similarity on L2-normalized embedding rows"""

from __future__ import annotations

import numpy as np


def cosine_top_k(
    corpus_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """
    corpus_embeddings: shape (n, d), rows should be L2-normalized.
    query_embedding: shape (d,), L2-normalized.
    Returns list of (row_index, similarity) sorted best-first. Similarity in [-1, 1] for normalized vectors.
    """

    if corpus_embeddings.ndim != 2 or query_embedding.ndim != 1:
        raise ValueError("corpus_embeddings must be 2d and query_embedding 1d")
    n, d = corpus_embeddings.shape
    if query_embedding.shape[0] != d:
        raise ValueError("dimension mismatch between corpus and query")
    if n == 0:
        return []

    scores = corpus_embeddings @ query_embedding
    k_eff = max(1, min(k, n))
    order = np.argsort(-scores)[:k_eff]
    return [(int(i), float(scores[i])) for i in order]

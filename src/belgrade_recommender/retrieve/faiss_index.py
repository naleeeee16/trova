"""On-disk FAISS index (IndexFlatIP) for L2-normalized embeddings, cosine via inner product."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_NAME = "manifest.json"
EVENT_IDS_NAME = "event_ids.json"
INDEX_NAME = "events.faiss"


def _require_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            'Install FAISS: pip install -e ".[retrieve]" (includes faiss-cpu)',
        ) from exc
    return faiss


def save_flat_ip_index(
    vectors: np.ndarray,
    event_ids: list[str],
    output_dir: Path,
    *,
    model_name: str,
) -> None:
    """
    Write ``events.faiss`` (IndexFlatIP), ``event_ids.json``, ``manifest.json``.

    ``vectors`` must be float32, shape (n, d), rows L2-normalized (same as encoder output).
    """

    faiss = _require_faiss()
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2d")
    n, d = vectors.shape
    if len(event_ids) != n:
        raise ValueError("event_ids length must match number of rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.ascontiguousarray(vectors.astype(np.float32))
    index = faiss.IndexFlatIP(int(d))
    index.add(x)
    faiss.write_index(index, str(output_dir / INDEX_NAME))
    (output_dir / EVENT_IDS_NAME).write_text(json.dumps(event_ids), encoding="utf-8")
    manifest = {
        "model_name": model_name,
        "dim": int(d),
        "n_vectors": int(n),
        "index_type": "IndexFlatIP",
    }
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_event_ids(output_dir: Path) -> list[str]:
    path = output_dir / EVENT_IDS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing event ids: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("event_ids.json must be a JSON array")
    return [str(x) for x in data]


def load_index(output_dir: Path) -> Any:
    faiss = _require_faiss()
    index_path = output_dir / INDEX_NAME
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    return faiss.read_index(str(index_path))


def search(
    output_dir: Path,
    query_vector: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """
    Return up to ``k`` pairs ``(row_index, inner_product)`` for L2-normalized query.

    Higher score is more similar (same as cosine for normalized vectors).
    """

    if query_vector.ndim != 1:
        raise ValueError("query_vector must be 1d")
    index = load_index(output_dir)
    n = int(index.ntotal)
    if n == 0:
        return []
    k_eff = max(1, min(int(k), n))
    q = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))
    scores, indices = index.search(q, k_eff)
    row_scores = scores[0]
    row_indices = indices[0]
    out: list[tuple[int, float]] = []
    for i in range(k_eff):
        idx = int(row_indices[i])
        if idx < 0:  # faiss padding when fewer than k results
            continue
        out.append((idx, float(row_scores[i])))
    return out

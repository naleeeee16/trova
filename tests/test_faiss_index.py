from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from belgrade_recommender.retrieve.faiss_index import (
    load_event_ids,
    load_manifest,
    save_flat_ip_index,
    search,
)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


def test_save_load_search_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((12, 32)).astype(np.float32)
    vectors = _l2_normalize_rows(raw)
    event_ids = [f"id_{i}" for i in range(12)]
    out = tmp_path / "vec"
    save_flat_ip_index(vectors, event_ids, out, model_name="test-model")

    manifest = load_manifest(out)
    assert manifest["dim"] == 32
    assert manifest["n_vectors"] == 12
    assert load_event_ids(out) == event_ids

    q = vectors[3].copy()
    hits = search(out, q, k=5)
    assert hits[0][0] == 3
    assert hits[0][1] > 0.99

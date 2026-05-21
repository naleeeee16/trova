"""
Default: embeddings for the whole loaded JSONL are computed in RAM inside ``retrieve/service.py``.
Fine for smoke slices.

**On-disk FAISS:``retrieve/faiss_index.py`` + ``scripts/build_vector_index.py`` →
``data/processed/vectors/events.faiss`` + ``event_ids.json`` + ``manifest.json``.
``retrieve/service.py`` accepts ``faiss_index_dir`` for semantic + hard / group flows.

**Later — Qdrant / Pinecone / pgvector:** metadata + ANN in one query; add e.g.
``retrieve/qdrant_store.py`` with the same conceptual roles (``build``, ``search``,
optional ``filter``).

"""

__all__: list[str] = []

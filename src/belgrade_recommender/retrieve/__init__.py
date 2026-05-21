"""Candidate retrieval: filters + vector / semantic search.

Import high-level helpers from submodules to avoid pulling numpy / sentence-transformers
on ``import belgrade_recommender.retrieve``:

- ``from belgrade_recommender.retrieve.text import build_embedding_text``
- ``from belgrade_recommender.retrieve.service import retrieve_top_k, retrieve_from_jsonl_file``
"""

from belgrade_recommender.retrieve.text import build_embedding_text

__all__ = ["build_embedding_text"]

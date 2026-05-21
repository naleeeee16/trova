"""High-level: embed corpus + query, return ranked events."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from belgrade_recommender.extract.attributes import extract_event_attributes_maybe_lazy_gemini
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.retrieve.cosine_search import cosine_top_k
from belgrade_recommender.retrieve.encoder import SentenceEncoder
from belgrade_recommender.retrieve.hard_filters import (
    passes_all_hard_constraints,
    passes_hard_constraints,
)
from belgrade_recommender.retrieve.text import build_embedding_text
from belgrade_recommender.schemas.parsed_preferences import ParsedHardConstraints


def events_by_id(events: Sequence[NormalizedEvent]) -> dict[str, NormalizedEvent]:
    return {e.event_id: e for e in events}


def load_normalized_events_jsonl(path: Path) -> list[NormalizedEvent]:
    out: list[NormalizedEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(NormalizedEvent.model_validate_json(line))
            except Exception:
                continue
    return out


def load_normalized_events(path: Path) -> list[NormalizedEvent]:
    """Load corpus from JSONL or from a SQLite DB (``.db`` / ``.sqlite``) built by ``load_jsonl_to_sqlite``."""

    suf = path.suffix.lower()
    if suf in (".db", ".sqlite", ".sqlite3"):
        from belgrade_recommender.storage.sqlite_store import load_all_normalized_events

        return load_all_normalized_events(path)
    return load_normalized_events_jsonl(path)


def retrieve_top_k(
    events: Sequence[NormalizedEvent],
    query_text: str,
    *,
    encoder: SentenceEncoder | None = None,
    top_k: int = 10,
) -> list[tuple[NormalizedEvent, float]]:
    """Return up to `top_k` events ranked by cosine similarity to `query_text`."""

    enc = encoder or SentenceEncoder()
    texts = [build_embedding_text(e) for e in events]
    corpus = enc.encode(texts)
    query_vec = enc.encode([query_text.strip()])[0]
    ranked = cosine_top_k(corpus, query_vec, top_k)
    return [(events[i], score) for i, score in ranked]


def retrieve_top_k_faiss(
    events_by_id: dict[str, NormalizedEvent],
    index_dir: Path,
    query_text: str,
    *,
    encoder: SentenceEncoder | None = None,
    top_k: int = 10,
) -> list[tuple[NormalizedEvent, float]]:
    """
    Nearest-neighbour search using a pre-built FAISS index (IndexFlatIP on normalized vectors).

    Only events whose ``event_id`` appears in ``events_by_id`` are returned (others skipped),
    e.g. when the JSONL slice is smaller than the indexed corpus.
    """

    from belgrade_recommender.retrieve import faiss_index

    enc = encoder or SentenceEncoder()
    manifest = faiss_index.load_manifest(index_dir)
    expected_model = manifest.get("model_name")
    if expected_model and expected_model != enc.model_name:
        raise ValueError(
            f"Index built with model {expected_model!r} but encoder is {enc.model_name!r}. "
            "Set SENTENCE_TRANSFORMER_MODEL to match or rebuild the index.",
        )
    query_vec = enc.encode([query_text.strip()])[0]
    hits = faiss_index.search(index_dir, query_vec, top_k)
    event_ids = faiss_index.load_event_ids(index_dir)
    out: list[tuple[NormalizedEvent, float]] = []
    for row_idx, score in hits:
        eid = event_ids[row_idx]
        ev = events_by_id.get(eid)
        if ev is not None:
            out.append((ev, score))
    return out


def _semantic_ranked(
    events: Sequence[NormalizedEvent],
    query_text: str,
    *,
    pool: int,
    encoder: SentenceEncoder | None,
    faiss_index_dir: Path | None,
) -> list[tuple[NormalizedEvent, float]]:
    if faiss_index_dir is not None:
        by_id = events_by_id(events)
        return retrieve_top_k_faiss(by_id, faiss_index_dir, query_text, encoder=encoder, top_k=pool)
    return retrieve_top_k(events, query_text, encoder=encoder, top_k=min(pool, len(events)))


def retrieve_from_jsonl_file(
    events_path: Path,
    query_text: str,
    *,
    top_k: int = 10,
    encoder: SentenceEncoder | None = None,
    faiss_index_dir: Path | None = None,
) -> list[tuple[NormalizedEvent, float]]:
    events = load_normalized_events(events_path)
    if faiss_index_dir is not None:
        return retrieve_top_k_faiss(
            events_by_id(events),
            faiss_index_dir,
            query_text,
            encoder=encoder,
            top_k=top_k,
        )
    return retrieve_top_k(events, query_text, encoder=encoder, top_k=top_k)


def retrieve_semantic_then_hard_filter(
    events: Sequence[NormalizedEvent],
    query_text: str,
    hard: ParsedHardConstraints,
    *,
    semantic_pool: int = 50,
    result_k: int = 10,
    encoder: SentenceEncoder | None = None,
    faiss_index_dir: Path | None = None,
) -> list[tuple[NormalizedEvent, float]]:
    """
    wide semantic pool, then **hard** constraint filtering

    ``EventAttributes`` are computed on the fly so we stay
    aligned with the same fields used for filtering later in SQL/DB.
    """

    pool = max(result_k, semantic_pool)
    ranked = _semantic_ranked(
        events,
        query_text,
        pool=pool,
        encoder=encoder,
        faiss_index_dir=faiss_index_dir,
    )
    out: list[tuple[NormalizedEvent, float]] = []
    for event, score in ranked:
        attrs = extract_event_attributes_maybe_lazy_gemini(event)
        if passes_hard_constraints(hard, event, attrs):
            out.append((event, score))
        if len(out) >= result_k:
            break
    return out


def retrieve_semantic_then_group_hard_filter(
    events: Sequence[NormalizedEvent],
    query_text: str,
    hards: Sequence[ParsedHardConstraints],
    *,
    semantic_pool: int = 80,
    result_k: int = 20,
    encoder: SentenceEncoder | None = None,
    faiss_index_dir: Path | None = None,
) -> list[tuple[NormalizedEvent, float]]:
    """Semantic pool, then keep only events that satisfy all members' hard constraints."""

    pool = max(result_k, semantic_pool)
    ranked = _semantic_ranked(
        events,
        query_text,
        pool=pool,
        encoder=encoder,
        faiss_index_dir=faiss_index_dir,
    )
    out: list[tuple[NormalizedEvent, float]] = []
    for event, score in ranked:
        attrs = extract_event_attributes_maybe_lazy_gemini(event)
        if passes_all_hard_constraints(hards, event, attrs):
            out.append((event, score))
        if len(out) >= result_k:
            break
    return out


def retrieve_from_jsonl_with_hard(
    events_path: Path,
    query_text: str,
    hard: ParsedHardConstraints,
    *,
    semantic_pool: int = 50,
    result_k: int = 10,
    encoder: SentenceEncoder | None = None,
    faiss_index_dir: Path | None = None,
) -> list[tuple[NormalizedEvent, float]]:
    events = load_normalized_events(events_path)
    return retrieve_semantic_then_hard_filter(
        events,
        query_text,
        hard,
        semantic_pool=semantic_pool,
        result_k=result_k,
        encoder=encoder,
        faiss_index_dir=faiss_index_dir,
    )

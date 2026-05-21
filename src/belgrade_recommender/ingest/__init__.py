"""Ingestion: load JSONL, resolve placeholders, normalize tags."""

from belgrade_recommender.ingest.heuristics import classify_for_retrieval_index
from belgrade_recommender.ingest.models import NormalizedEvent, RawEventRecord
from belgrade_recommender.ingest.pipeline import ingest_jsonl_file, iter_normalized_events

__all__ = [
    "NormalizedEvent",
    "RawEventRecord",
    "classify_for_retrieval_index",
    "ingest_jsonl_file",
    "iter_normalized_events",
]

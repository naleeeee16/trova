"""Build a single string per event for semantic embedding."""

from __future__ import annotations

from belgrade_recommender.ingest.models import NormalizedEvent


def build_embedding_text(event: NormalizedEvent, *, max_chars: int = 2500) -> str:
    """
    Concatenate stable fields so the vector reflects channel, tags and English text.
    Uses resolved English description;
    """

    tags = ", ".join(event.tags) if event.tags else ""
    body = (event.event_description_resolved or "").strip()
    parts = [
        f"channel: {event.source_channel}",
        f"tags: {tags}",
        f"description: {body}",
    ]
    text = "\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text

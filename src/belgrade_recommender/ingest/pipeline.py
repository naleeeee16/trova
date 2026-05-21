from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from belgrade_recommender.ingest.heuristics import classify_for_retrieval_index
from belgrade_recommender.ingest.ids import stable_event_id
from belgrade_recommender.ingest.models import NormalizedEvent, RawEventRecord
from belgrade_recommender.ingest.placeholders import resolve_placeholders
from belgrade_recommender.ingest.tags import normalize_tags

log = logging.getLogger(__name__)


def parse_jsonl_line(row_index: int, line: str) -> NormalizedEvent | None:
    """Parse one non-empty JSONL line into NormalizedEvent, or None if skipped."""

    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    try:
        raw = RawEventRecord.model_validate(data)
    except Exception:
        return None
    if not raw.link:
        return None
    links = raw.links if isinstance(raw.links, dict) else {}
    tags = normalize_tags(raw.tags)
    desc_resolved = resolve_placeholders(raw.event_description, links)
    include, reason = classify_for_retrieval_index(
        event_description_resolved=desc_resolved,
        tags=tags,
        event_description_raw=raw.event_description,
    )
    return NormalizedEvent(
        event_id=stable_event_id(raw.link),
        row_index=row_index,
        link=raw.link,
        source_channel=raw.source_channel,
        tags=tags,
        event_description_raw=raw.event_description,
        event_description_resolved=desc_resolved,
        ru_event_description_raw=raw.ru_event_description,
        ru_event_description_resolved=resolve_placeholders(raw.ru_event_description, links),
        include_in_index=include,
        exclude_reason=reason,
    )


def iter_normalized_events(path: Path) -> Iterator[NormalizedEvent]:
    """Stream normalized events from a JSONL file. Skips invalid lines."""

    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            event = parse_jsonl_line(row_index, line)
            if event is not None:
                yield event


def ingest_jsonl_file(
    input_path: Path,
    output_path: Path,
    *,
    classify_events: bool = False,
    classifier_batch_size: int = 20,
) -> tuple[int, int]:
    """
    Read input JSONL, write normalized JSONL (one JSON object per line).

    Returns (written_count, skipped_count) where skipped includes empty lines and bad rows.

    If classify_events=True, events that passed the heuristic filter are sent
    to an LLM in batches to confirm they are attendable public events.  Those
    classified as non-events get include_in_index=False / exclude_reason="llm:non_event".
    Requires OPENAI_API_KEY to be set.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: parse all lines into NormalizedEvent objects.
    events: list[NormalizedEvent] = []
    skipped = 0
    with input_path.open(encoding="utf-8") as inp:
        for row_index, line in enumerate(inp):
            if not line.strip():
                skipped += 1
                continue
            event = parse_jsonl_line(row_index, line)
            if event is None:
                skipped += 1
                continue
            events.append(event)

    # Optional second pass: LLM classification of heuristically-included events.
    if classify_events:
        from belgrade_recommender.llm.openai_event_classifier import classify_events_batch

        candidates = [e for e in events if e.include_in_index]
        log.info(
            "LLM event classification: sending %d events to OpenAI (batch_size=%d)…",
            len(candidates),
            classifier_batch_size,
        )
        descriptions = [e.event_description_resolved or e.event_description_raw for e in candidates]
        is_event_flags = classify_events_batch(descriptions, batch_size=classifier_batch_size)

        excluded = 0
        for ev, is_event in zip(candidates, is_event_flags):
            if not is_event:
                ev.include_in_index = False
                ev.exclude_reason = "llm:non_event"
                excluded += 1

        log.info(
            "LLM classification done: %d/%d marked as non-events.",
            excluded,
            len(candidates),
        )

    # Write all events (included + excluded) to output so the file is a
    # complete record; the index loader filters on include_in_index at load time.
    with output_path.open("w", encoding="utf-8") as out:
        for event in events:
            out.write(event.model_dump_json() + "\n")

    written = len(events)
    return written, skipped

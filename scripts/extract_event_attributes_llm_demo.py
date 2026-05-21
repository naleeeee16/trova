#!/usr/bin/env python3
"""Demo: Gemini structured JSON for one event's attributes (compare with rule-based extract)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.extract.attributes import extract_event_attributes
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.llm.gemini_event_attributes import parse_event_attributes_with_gemini
from belgrade_recommender.retrieve.service import load_normalized_events_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini vs rules event attributes (one event).")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("fixtures/events_smoke.jsonl"),
        help="JSONL to read first valid row from.",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        default="",
        help="If set, pick this event_id from the file; else use first row.",
    )
    args = parser.parse_args()

    if not args.events.is_file():
        print(f"File not found: {args.events}", file=sys.stderr)
        sys.exit(1)

    rows = load_normalized_events_jsonl(args.events)
    if not rows:
        print("No events in file.", file=sys.stderr)
        sys.exit(1)

    event: NormalizedEvent | None = None
    if args.event_id.strip():
        event = next((e for e in rows if e.event_id == args.event_id.strip()), None)
    else:
        event = rows[0]
    if event is None:
        print("Event not found.", file=sys.stderr)
        sys.exit(1)

    rules = extract_event_attributes(event)
    print("=== rules_v1 ===")
    print(rules.model_dump_json(indent=2))

    try:
        gem = parse_event_attributes_with_gemini(event)
    except (ImportError, RuntimeError, ValueError) as exc:
        print("\n=== gemini_v1 (failed) ===", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print("\n=== gemini_v1 ===")
    print(gem.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

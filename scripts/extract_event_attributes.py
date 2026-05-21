#!/usr/bin/env python3
"""Phase 2: append rule-based EventAttributes for each normalized JSONL row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.extract.attributes import extract_event_attributes
from belgrade_recommender.ingest.models import NormalizedEvent


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured event attributes (rules v1).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/events_normalized.jsonl"),
        help="Normalized JSONL from ingest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/events_structured.jsonl"),
        help="Output: one JSON per line with event_id + attributes.",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input.open(encoding="utf-8") as inp, args.output.open("w", encoding="utf-8") as out:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                event = NormalizedEvent.model_validate_json(line)
            except Exception:
                continue
            attrs = extract_event_attributes(event)
            row = {"event_id": event.event_id, "attributes": attrs.model_dump()}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()

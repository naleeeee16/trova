#!/usr/bin/env python3
"""Load ``events_normalized.jsonl`` into SQLite (relational + rule-based ``event_attributes``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.storage.sqlite_store import bulk_load_normalized_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite DB from normalized events JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/events_normalized.jsonl"),
        help="Source JSONL of NormalizedEvent rows.",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("data/processed/events.sqlite"),
        help="Target SQLite file (created or overwritten).",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="If set, do not clear tables before load (uses fresh inserts; may fail on duplicate keys).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    n = bulk_load_normalized_jsonl(args.input, args.output_db, replace=not args.no_replace)
    print(f"Inserted {n} events into {args.output_db}")


if __name__ == "__main__":
    main()

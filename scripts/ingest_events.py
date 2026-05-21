"""CLI: normalize events JSONL (Phase 1 ingest)."""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Allow running without editable install
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.ingest.pipeline import ingest_jsonl_file


def main() -> None:
    default_input = os.environ.get("EVENTS_JSONL_PATH", "").strip()
    parser = argparse.ArgumentParser(description="Normalize events JSONL for Phase 1 ingest.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(default_input) if default_input else Path("data/raw/events.jsonl"),
        help="Path to events_dataset.jsonl (or set EVENTS_JSONL_PATH).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/events_normalized.jsonl"),
        help="Output normalized JSONL path.",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    written, skipped = ingest_jsonl_file(args.input, args.output)
    print(f"Wrote {written} rows to {args.output} (skipped {skipped} lines)")


if __name__ == "__main__":
    main()

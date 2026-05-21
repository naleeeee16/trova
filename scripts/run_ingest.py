#!/usr/bin/env python3
"""
One-shot ingestion script for the Belgrade event corpus.

Reads the raw JSONL dataset, applies heuristic filters, and optionally runs
LLM-based event classification to remove catering ads, blogger videos, and
other non-attendable content.

Usage:
    # Basic ingest (heuristics only, fast, free):
    python run_ingest.py

    # Full LLM classification (recommended before production deployment):
    python run_ingest.py --classify-events

    # Custom paths:
    python run_ingest.py --dataset fixtures/events_dataset.jsonl --classify-events

After running this, all test scripts and the bot can use --skip-ingest to
reuse the normalized file without re-running ingestion.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC  = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_env_file = _ROOT / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and normalize the Belgrade event corpus.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_ROOT / "fixtures" / "events_dataset.jsonl",
        help="Path to raw events JSONL (default: fixtures/events_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "processed" / "events_normalized_test.jsonl",
        help="Path to write normalized JSONL (default: data/processed/events_normalized_test.jsonl)",
    )
    parser.add_argument(
        "--classify-events",
        action="store_true",
        help=(
            "Run LLM classification on all heuristically-included events to remove "
            "catering ads, blogger videos, and other non-attendable content. "
            "Requires OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Number of events per LLM batch (default: 15)",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        log.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    if args.classify_events and not os.environ.get("OPENAI_API_KEY"):
        log.error("--classify-events requires OPENAI_API_KEY to be set.")
        sys.exit(1)

    from belgrade_recommender.ingest.pipeline import ingest_jsonl_file

    log.info("Ingesting %s …", args.dataset)
    t0 = time.perf_counter()

    written, skipped = ingest_jsonl_file(
        args.dataset,
        args.output,
        classify_events=args.classify_events,
        classifier_batch_size=args.batch_size,
    )

    elapsed = time.perf_counter() - t0
    log.info("Done in %.1fs — %d written, %d skipped", elapsed, written, skipped)
    log.info("Output: %s", args.output)

    if args.classify_events:
        # Show how many ended up excluded by LLM
        from belgrade_recommender.retrieve.service import load_normalized_events
        all_events = load_normalized_events(args.output)
        llm_excluded = sum(
            1 for e in all_events
            if not e.include_in_index and e.exclude_reason == "llm:non_event"
        )
        total_included = sum(1 for e in all_events if e.include_in_index)
        log.info(
            "LLM removed %d non-events. %d events remain in index.",
            llm_excluded,
            total_included,
        )


if __name__ == "__main__":
    main()

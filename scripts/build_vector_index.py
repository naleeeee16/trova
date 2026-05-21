#!/usr/bin/env python3
"""Build on-disk FAISS index + event id list from normalized events JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.retrieve.encoder import SentenceEncoder
from belgrade_recommender.retrieve.faiss_index import save_flat_ip_index
from belgrade_recommender.retrieve.service import load_normalized_events
from belgrade_recommender.retrieve.text import build_embedding_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode events and write FAISS index under data/processed/vectors/.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/events_normalized.jsonl"),
        help="Normalized events JSONL (same format as ingest output).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/vectors"),
        help="Directory for events.faiss, event_ids.json, manifest.json",
    )
    parser.add_argument(
        "--only-indexable",
        action="store_true",
        help="Only rows with include_in_index=true (recommended for retrieval corpus).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows: list[NormalizedEvent] = load_normalized_events(args.input)
    if args.only_indexable:
        rows = [e for e in rows if e.include_in_index]
    if not rows:
        print("No events to index after filtering.", file=sys.stderr)
        sys.exit(1)

    texts = [build_embedding_text(e) for e in rows]
    event_ids = [e.event_id for e in rows]

    try:
        enc = SentenceEncoder()
        vectors = enc.encode(texts)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    save_flat_ip_index(vectors, event_ids, args.output_dir, model_name=enc.model_name)
    print(f"Wrote index for n={len(rows)} dim={vectors.shape[1]} model={enc.model_name!r} -> {args.output_dir}")


if __name__ == "__main__":
    main()

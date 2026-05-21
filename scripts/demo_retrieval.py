#!/usr/bin/env python3
"""Demo: semantic top-k over a normalized JSONL slice (for example fixtures/events_smoke.jsonl)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.retrieve.service import (
    retrieve_from_jsonl_file,
    retrieve_from_jsonl_with_hard,
)
from belgrade_recommender.schemas.parsed_preferences import ParsedHardConstraints


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic retrieval demo (sentence-transformers + cosine).")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("fixtures/events_smoke.jsonl"),
        help="JSONL of NormalizedEvent rows.",
    )
    parser.add_argument("--query", type=str, default="", help="Natural-language query.")
    parser.add_argument(
        "--fixture-user",
        type=str,
        default="",
        help="If set and --query empty: use preferences_plain_text from fixtures/synthetic_users.json.",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="After a wider semantic pool, drop events that violate fixture user's hard_constraints (requires --fixture-user).",
    )
    parser.add_argument("--semantic-pool", type=int, default=60, help="With --hard: how many semantic candidates to scan.")
    parser.add_argument(
        "--faiss-index",
        type=Path,
        default=None,
        help="Directory with events.faiss + event_ids.json + manifest.json (semantic search via FAISS).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    query = args.query.strip()
    hard: ParsedHardConstraints | None = None
    if not query and args.fixture_user:
        data = json.loads((_ROOT / "fixtures" / "synthetic_users.json").read_text(encoding="utf-8"))
        users = {u["user_id"]: u for u in data["users"]}
        u = users.get(args.fixture_user)
        if not u:
            print(f"Unknown user_id: {args.fixture_user}", file=sys.stderr)
            sys.exit(1)
        query = u["preferences_plain_text"]
        if args.hard:
            hard = ParsedHardConstraints.model_validate(u["hard_constraints"])

    if not query:
        parser.error("Provide --query or --fixture-user")

    if args.hard and hard is None:
        parser.error("--hard requires --fixture-user (hard_constraints loaded from fixture).")

    if not args.events.is_file():
        print(f"Events file not found: {args.events}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.hard and hard is not None:
            ranked = retrieve_from_jsonl_with_hard(
                args.events,
                query,
                hard,
                semantic_pool=args.semantic_pool,
                result_k=args.top_k,
                faiss_index_dir=args.faiss_index,
            )
        else:
            ranked = retrieve_from_jsonl_file(
                args.events,
                query,
                top_k=args.top_k,
                faiss_index_dir=args.faiss_index,
            )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    mode = "semantic+hard" if args.hard else "semantic"
    if args.faiss_index:
        mode += "+faiss"
    print(f"mode={mode}  results={len(ranked)}\n")
    for rank, (event, score) in enumerate(ranked, start=1):
        title = (event.event_description_resolved or "")[:120].replace("\n", " ")
        print(f"{rank:2d}  score={score:.4f}  {event.event_id}")
        print(f"    {event.link}")
        print(f"    {title}...")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demo: group query - semantic pool - intersection of hard filters - least-misery soft ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.rank.group_rank import (
    combined_group_query_text,
    fixture_user_to_prefs,
    rank_by_least_misery,
)
from belgrade_recommender.retrieve.service import (
    load_normalized_events,
    retrieve_semantic_then_group_hard_filter,
)
from belgrade_recommender.schemas.parsed_preferences import ParsedHardConstraints


def main() -> None:
    parser = argparse.ArgumentParser(description="Group retrieval + least-misery ranking demo.")
    parser.add_argument("--events", type=Path, default=Path("fixtures/events_smoke.jsonl"))
    parser.add_argument(
        "--group",
        type=str,
        default="weekend_group",
        help="group_id from fixtures/synthetic_users.json",
    )
    parser.add_argument("--semantic-pool", type=int, default=80)
    parser.add_argument("--after-hard-k", type=int, default=25, help="Max candidates after group hard filter.")
    parser.add_argument(
        "--faiss-index",
        type=Path,
        default=None,
        help="Directory with events.faiss + event_ids.json (semantic pool via FAISS).",
    )
    parser.add_argument("--top", type=int, default=10, help="How many to print after group ranking.")
    args = parser.parse_args()

    fixture_path = _ROOT / "fixtures" / "synthetic_users.json"
    if not fixture_path.is_file():
        print(f"Fixture not found: {fixture_path}", file=sys.stderr)
        sys.exit(1)
    if not args.events.is_file():
        print(f"Events file not found: {args.events}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    users_by_id = {u["user_id"]: u for u in data["users"]}
    group = next((g for g in data["groups"] if g["group_id"] == args.group), None)
    if not group:
        print(f"Unknown group_id: {args.group}", file=sys.stderr)
        sys.exit(1)

    member_users: list[dict] = []
    for uid in group["member_user_ids"]:
        u = users_by_id.get(uid)
        if u is None:
            print(f"Group references unknown user_id: {uid}", file=sys.stderr)
            sys.exit(1)
        member_users.append(u)

    member_prefs = [fixture_user_to_prefs(u) for u in member_users]
    hards = [ParsedHardConstraints.model_validate(u["hard_constraints"]) for u in member_users]
    query = combined_group_query_text(member_users)

    try:
        events = load_normalized_events(args.events)
        candidates = retrieve_semantic_then_group_hard_filter(
            events,
            query,
            hards,
            semantic_pool=args.semantic_pool,
            result_k=args.after_hard_k,
            faiss_index_dir=args.faiss_index,
        )
        ranked = rank_by_least_misery(candidates, member_prefs)[: args.top]
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    mode = "group"
    if args.faiss_index:
        mode += "+faiss"
    print(f"{mode}  group={args.group}  members={group['member_user_ids']}")
    print(f"candidates_after_hard={len(candidates)}  showing_top={len(ranked)}\n")
    for i, (event, sem, min_s, mean_s) in enumerate(ranked, start=1):
        title = (event.event_description_resolved or "")[:120].replace("\n", " ")
        print(f"{i:2d}  sem={sem:.4f}  min_soft={min_s:.4f}  mean_soft={mean_s:.4f}  {event.event_id}")
        print(f"    {event.link}")
        print(f"    {title}...")
        print()


if __name__ == "__main__":
    main()

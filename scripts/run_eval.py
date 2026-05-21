#!/usr/bin/env python3
"""Offline eval: P@k / R@k / MRR / NDCG from fixtures/eval_labels.json.

Runs each case in two modes automatically (ablation):
  semantic-only  — pure cosine similarity, no constraint filtering
  semantic+hard  — cosine pool then hard-constraint filtering

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --spec fixtures/eval_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.eval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from belgrade_recommender.retrieve.service import (
    retrieve_from_jsonl_file,
    retrieve_from_jsonl_with_hard,
)
from belgrade_recommender.schemas.parsed_preferences import ParsedHardConstraints


def _run_case(
    case: dict,
    users: dict,
    *,
    mode: str,  # "semantic" | "hard"
) -> dict | None:
    """Return metric dict for one case in the given mode, or None if skipped."""
    cid = case.get("case_id", "?")
    events_path = Path(case["events_path"])
    if not events_path.is_file():
        print(f"[{cid}] skip: events file not found: {events_path}", file=sys.stderr)
        return None

    uid = case["fixture_user_id"]
    u = users.get(uid)
    if not u:
        print(f"[{cid}] skip: unknown user {uid}", file=sys.stderr)
        return None

    relevant = set(case.get("relevant_event_ids", []))
    if not relevant:
        print(f"[{cid}] skip: empty relevant_event_ids", file=sys.stderr)
        return None

    k = int(case.get("k", 10))
    pool = int(case.get("semantic_pool", max(60, k * 5)))

    try:
        if mode == "hard":
            hard = ParsedHardConstraints.model_validate(u["hard_constraints"])
            ranked = retrieve_from_jsonl_with_hard(
                events_path,
                u["preferences_plain_text"],
                hard,
                semantic_pool=pool,
                result_k=k,
            )
        else:
            ranked = retrieve_from_jsonl_file(
                events_path,
                u["preferences_plain_text"],
                top_k=k,
            )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    ranked_ids = [ev.event_id for ev, _ in ranked]
    return {
        "case_id": cid,
        "k": k,
        "n_retrieved": len(ranked_ids),
        "p": precision_at_k(relevant, ranked_ids, k),
        "r": recall_at_k(relevant, ranked_ids, k),
        "rr": reciprocal_rank(relevant, ranked_ids),
        "ndcg": ndcg_at_k(relevant, ranked_ids, k),
    }


def _macro(results: list[dict]) -> dict:
    keys = ("p", "r", "rr", "ndcg")
    return {key: sum(r[key] for r in results) / len(results) for key in keys}


def _print_row(label: str, m: dict, k: int | None = None) -> None:
    k_str = f"@{k:<2d}" if k is not None else "   "
    print(
        f"{label:<32s}  P{k_str}={m['p']:.3f}  R{k_str}={m['r']:.3f}"
        f"  MRR={m['rr']:.3f}  NDCG={m['ndcg']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("fixtures/eval_labels.json"))
    parser.add_argument("--fixture-users", type=Path, default=Path("fixtures/synthetic_users.json"))
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Skip semantic-only mode; run hard-filter mode only",
    )
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"Spec not found: {args.spec}", file=sys.stderr)
        sys.exit(1)
    if not args.fixture_users.is_file():
        print(f"Fixture users not found: {args.fixture_users}", file=sys.stderr)
        sys.exit(1)

    bundle = json.loads(args.spec.read_text(encoding="utf-8"))
    cases = bundle.get("cases", [])
    if not cases:
        print("No cases in spec.", file=sys.stderr)
        sys.exit(1)

    users_data = json.loads(args.fixture_users.read_text(encoding="utf-8"))
    users = {u["user_id"]: u for u in users_data.get("users", [])}

    sem_results: list[dict] = []
    hard_results: list[dict] = []

    header = f"{'case_id':<32s}  {'semantic-only':>42s}  {'semantic+hard':>42s}"
    divider = "-" * len(header)

    if not args.no_ablation:
        print("\n=== Ablation: semantic-only vs. semantic+hard ===\n")
        print(f"{'case_id':<32s}  {'--- semantic-only ---':^42s}  {'--- semantic+hard ---':^42s}")
        print(f"{'':32s}  {'P@k   R@k   MRR   NDCG':^42s}  {'P@k   R@k   MRR   NDCG':^42s}")
        print(divider)

    for case in cases:
        sem = _run_case(case, users, mode="semantic")
        hard = _run_case(case, users, mode="hard")

        if sem is None or hard is None:
            continue

        sem_results.append(sem)
        hard_results.append(hard)

        if not args.no_ablation:
            k = sem["k"]
            sem_str = f"P@{k:<2d}={sem['p']:.3f}  R={sem['r']:.3f}  MRR={sem['rr']:.3f}  NDCG={sem['ndcg']:.3f}"
            hard_str = f"P@{k:<2d}={hard['p']:.3f}  R={hard['r']:.3f}  MRR={hard['rr']:.3f}  NDCG={hard['ndcg']:.3f}"
            print(f"{sem['case_id']:<32s}  {sem_str:<42s}  {hard_str}")

    if not sem_results:
        print("No cases evaluated.", file=sys.stderr)
        sys.exit(1)

    sem_avg = _macro(sem_results)
    hard_avg = _macro(hard_results)

    if not args.no_ablation:
        print(divider)
        sem_str = f"P={sem_avg['p']:.3f}  R={sem_avg['r']:.3f}  MRR={sem_avg['rr']:.3f}  NDCG={sem_avg['ndcg']:.3f}"
        hard_str = f"P={hard_avg['p']:.3f}  R={hard_avg['r']:.3f}  MRR={hard_avg['rr']:.3f}  NDCG={hard_avg['ndcg']:.3f}"
        print(f"{'macro_avg':<32s}  {sem_str:<42s}  {hard_str}")

        ndcg_delta = hard_avg["ndcg"] - sem_avg["ndcg"]
        mrr_delta = hard_avg["rr"] - sem_avg["rr"]
        print(f"\nHard-filter delta:  ΔNDCG={ndcg_delta:+.3f}  ΔMRR={mrr_delta:+.3f}  "
              f"({'hard filter helps' if ndcg_delta > 0 else 'hard filter hurts on this corpus'})")

    print(f"\n=== semantic+hard (final pipeline) — {len(hard_results)} cases ===\n")
    for r in hard_results:
        _print_row(r["case_id"], r, k=r["k"])
    print()
    _print_row(f"macro_avg  n={len(hard_results)}", hard_avg)


if __name__ == "__main__":
    main()

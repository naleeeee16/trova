#!/usr/bin/env python3
"""Build a small JSONL slice from normalized events for fast tests (fixtures/events_smoke.jsonl)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

def main() -> None:
    parser = argparse.ArgumentParser(description="Sample normalized JSONL into fixtures/events_smoke.jsonl.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/events_normalized.jsonl"),
        help="Normalized JSONL from ingest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/events_smoke.jsonl"),
        help="Output smoke JSONL path.",
    )
    parser.add_argument("--sample-size", type=int, default=150, help="Number of rows to write.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling.")
    parser.add_argument(
        "--include-non-indexable",
        action="store_true",
        help="Also allow rows with include_in_index=false (for edge-case tests). Default: only indexable rows.",
    )
    args = parser.parse_args()
    indexable_only = not args.include_non_indexable

    if not args.input.is_file():
        print(f"Input not found: {args.input} — run ingest first.", file=sys.stderr)
        sys.exit(1)

    pool: list[str] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if indexable_only:
                try:
                    row = json.loads(line)
                    if not row.get("include_in_index", True):
                        continue
                except json.JSONDecodeError:
                    continue
            pool.append(line)

    if not pool:
        print("No rows in pool after filtering.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    k = min(args.sample_size, len(pool))
    chosen = rng.sample(pool, k=k) if k < len(pool) else pool

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for line in chosen:
            out.write(line + "\n")

    print(f"Wrote {len(chosen)} rows to {args.output} (pool size {len(pool)}, indexable_only={indexable_only})")


if __name__ == "__main__":
    main()

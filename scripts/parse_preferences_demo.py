#!/usr/bin/env python3
"""Demo: parse one user's plain-text preferences via Gemini structured JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from belgrade_recommender.llm.gemini_preferences import parse_preferences_plain_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse preferences_plain_text with Gemini structured output.")
    parser.add_argument("text", nargs="?", help="Preference text. If omitted, use first fixture user.")
    parser.add_argument(
        "--fixture-user",
        type=str,
        default="ana",
        help="If text omitted: user_id from fixtures/synthetic_users.json (default: ana).",
    )
    args = parser.parse_args()

    text = args.text
    if not text:
        path = _ROOT / "fixtures" / "synthetic_users.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        users = {u["user_id"]: u for u in data["users"]}
        user = users.get(args.fixture_user)
        if not user:
            print(f"Unknown user_id: {args.fixture_user}", file=sys.stderr)
            sys.exit(1)
        text = user["preferences_plain_text"]

    try:
        parsed = parse_preferences_plain_text(text)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(parsed.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

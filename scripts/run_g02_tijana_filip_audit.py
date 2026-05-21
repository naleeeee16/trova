#!/usr/bin/env python3
"""
G02 Tijana + Filip combined group — tag-audit style output.

Runs Tijana and Filip as ONE group through the recommender pipeline and
shows per-event: raw tags, type_hints, matched/missed liked_types (same
format as run_tag_audit.py), plus the standard metrics lines that
run_all_tests.py can parse.

Usage:
    python run_g02_tijana_filip_audit.py
    python run_g02_tijana_filip_audit.py --skip-ingest
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
elif sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_env_file = _ROOT / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

import numpy as np

from belgrade_recommender.extract.attributes import extract_event_attributes_maybe_lazy_gemini
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.ingest.pipeline import ingest_jsonl_file
from belgrade_recommender.rank.group_rank import rank_by_least_misery
from belgrade_recommender.retrieve.cosine_search import cosine_top_k
from belgrade_recommender.retrieve.encoder import SentenceEncoder
from belgrade_recommender.retrieve.hard_filters import passes_all_hard_constraints
from belgrade_recommender.retrieve.service import load_normalized_events
from belgrade_recommender.retrieve.text import build_embedding_text
from belgrade_recommender.schemas.parsed_preferences import (
    ParsedHardConstraints,
    ParsedSoftPreferences,
    ParsedUserPreferences,
)

_SEP  = "=" * 72
_THIN = "-" * 72


def _member(
    name: str,
    query: str,
    *,
    liked_types: list[str] | None = None,
    noise: str = "unknown",
    must_be_free: bool | None = None,
) -> dict:
    prefs = ParsedUserPreferences(
        summary_one_line=name,
        hard_constraints=ParsedHardConstraints(
            must_be_free=must_be_free,
            forbidden_event_categories=[],
        ),
        soft_preferences=ParsedSoftPreferences(
            liked_types=liked_types or [],
            noise_tolerance=noise,  # type: ignore[arg-type]
        ),
    )
    return {"name": name, "query": query, "prefs": prefs}


GROUP = {
    "group_id":    "G02_studenti_bez_para",
    "description": "Tijana + Filip — sve mora biti besplatno",
    "members": [
        _member(
            "Tijana",
            query="nemam para uopste, trazim nesto besplatno za vikend u Beogradu",
            liked_types=["outdoor", "festival", "market", "social"],
            noise="medium",
            must_be_free=True,
        ),
        _member(
            "Filip",
            query="student sam, mogu samo na besplatne dogadjaje, volim muziku i umetnost",
            liked_types=["concert", "art", "exhibition", "music"],
            noise="medium_high",
            must_be_free=True,
        ),
    ],
}


def _safe(text: str | None, n: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


def _format_member_section(
    member: dict,
    ev: NormalizedEvent,
    attrs,
    sem: float,
    mn: float,
    mean: float,
    rank: int,
) -> list[str]:
    prefs: ParsedUserPreferences = member["prefs"]
    liked = list(prefs.soft_preferences.liked_types)
    liked_set = {t.lower() for t in liked}

    raw_tags   = ev.tags or []
    type_hints = list(getattr(attrs, "type_hints", [])) if attrs else []

    th_set  = {h.lower() for h in type_hints} | {t.lower() for t in raw_tags}
    matched = [lt for lt in liked if any(lt in x or x in lt for x in th_set)]
    missed  = [lt for lt in liked if lt not in matched]

    lines: list[str] = []
    lines.append(f"    [{member['name']}]  score={mn*100:.0f}%  (sem={sem:.3f} min={mn:.3f} mean={mean:.3f})")

    if raw_tags:
        lines.append(f"    raw tags   : {'  '.join(raw_tags[:15])}")
    else:
        lines.append("    raw tags   : (none)")

    if type_hints:
        lines.append(f"    type_hints : {',  '.join(type_hints)}")
    else:
        lines.append("    type_hints : (none)")

    if attrs:
        lines.append(
            f"    attrs      : noise={attrs.noise_level_hint}  "
            f"outdoor={attrs.outdoor_hint}  city={attrs.city_hint}  "
            f"free={attrs.price_free_signal}  paid={attrs.price_paid_signal}"
            + (f"  rsd={attrs.price_amount_rsd}" if attrs.price_amount_rsd else "")
            + (f"  eur={attrs.price_amount_eur}" if attrs.price_amount_eur else "")
        )

    if liked:
        if matched:
            lines.append(f"    ✓ matched  : {matched}")
        if missed:
            lines.append(f"    ✗ missed   : {missed}")
        if not matched and not missed:
            lines.append("    (no liked_types to match)")
    else:
        lines.append("    (user has no liked_types)")

    return lines


def run(
    events: list[NormalizedEvent],
    corpus_matrix: np.ndarray,
    encoder: SentenceEncoder,
) -> None:
    members = GROUP["members"]
    hards   = [m["prefs"].hard_constraints for m in members]
    prefs   = [m["prefs"] for m in members]

    joint_query = "\n\n".join(m["query"] for m in members)

    print(f"\n{_SEP}")
    print(f"  GROUP : {GROUP['group_id']}")
    print(f"          {GROUP['description']}")
    print(f"  Members : {', '.join(m['name'] for m in members)}")
    print(_THIN)
    for m in members:
        p: ParsedUserPreferences = m["prefs"]
        print(f"  {m['name']:<10}  liked={list(p.soft_preferences.liked_types)}  "
              f"noise={p.soft_preferences.noise_tolerance}  "
              f"free={p.hard_constraints.must_be_free}")
    print(_THIN)
    print(f"  Query (joint):")
    for m in members:
        print(f"    [{m['name']}] {m['query']}")
    print()

    t0 = time.perf_counter()

    query_vec = encoder.encode([joint_query.strip()])[0]
    pool_hits = cosine_top_k(corpus_matrix, query_vec, k=80)

    attrs_cache: dict = {}
    candidates: list[tuple[NormalizedEvent, float]] = []
    for idx, score in pool_hits:
        ev = events[idx]
        attrs = extract_event_attributes_maybe_lazy_gemini(ev)
        attrs_cache[ev.event_id] = attrs
        if passes_all_hard_constraints(hards, ev, attrs):
            candidates.append((ev, score))
        if len(candidates) >= 20:
            break

    ranked = rank_by_least_misery(candidates, prefs)[:10]
    latency = time.perf_counter() - t0

    hard_pass = sum(
        1 for ev, *_ in ranked
        if passes_all_hard_constraints(hards, ev, attrs_cache.get(ev.event_id))
    )
    hard_total = len(ranked)

    print(f"Latency    : {latency:.3f}s")
    print(f"Candidates : {len(candidates)}  (after semantic + hard filter)")
    print(f"Ranked     : {len(ranked)}")

    if hard_total > 0:
        rate = 100 * hard_pass / hard_total
        flag = "OK" if rate == 100 else "WARN"
        print(f"Hard constraints : {hard_pass}/{hard_total} ({rate:.0f}%)  [{flag}]")

    if not ranked:
        print(f"\n  WARNING: No results returned.")
        print(_SEP)
        return

    print(f"\n  RANKED RESULTS (tag-audit format):\n")

    for i, (ev, sem, mn, mean) in enumerate(ranked, 1):
        attrs = attrs_cache.get(ev.event_id)

        print(f"  {i}. {_safe(ev.event_description_resolved, 70)}")
        if ev.link:
            print(f"     link : {ev.link}")

        for member in members:
            lines = _format_member_section(member, ev, attrs, sem, mn, mean, i)
            for line in lines:
                print(line)
        print()

    print(_SEP)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=Path, default=_ROOT / "fixtures" / "events_dataset.jsonl")
    parser.add_argument("--normalized-out", type=Path, default=_ROOT / "data" / "processed" / "events_normalized_test.jsonl")
    parser.add_argument("--skip-ingest",    action="store_true")
    args = parser.parse_args()

    if not args.skip_ingest:
        if not args.dataset.is_file():
            print(f"ERROR: dataset not found: {args.dataset}"); sys.exit(1)
        print(f"Ingesting {args.dataset} …")
        written, skipped = ingest_jsonl_file(args.dataset, args.normalized_out)
        print(f"  {written} written, {skipped} skipped")
    else:
        if not args.normalized_out.is_file():
            print(f"ERROR: normalized file not found: {args.normalized_out}"); sys.exit(1)

    all_events = load_normalized_events(args.normalized_out)
    indexable  = [e for e in all_events if e.include_in_index]

    encoder = SentenceEncoder()
    corpus_matrix = encoder.encode([build_embedding_text(e) for e in indexable])

    run(indexable, corpus_matrix, encoder)


if __name__ == "__main__":
    main()

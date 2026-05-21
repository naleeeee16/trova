#!/usr/bin/env python3
"""
Tag audit: for every member query across all 4 test suites, show
raw event hashtags, derived type_hints, and liked_types match/miss.

Output: run_tag_audit_output.txt

Usage:
    python run_tag_audit.py
    python run_tag_audit.py --skip-ingest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import io
import os
import sys
# Force UTF-8 stdout on Windows so Serbian/Russian names don't crash progress prints
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
log = logging.getLogger("tagaudit")

import numpy as np

from belgrade_recommender.extract.attributes import extract_event_attributes_maybe_lazy_gemini
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.ingest.pipeline import ingest_jsonl_file
from belgrade_recommender.rank.group_rank import rank_by_least_misery
from belgrade_recommender.retrieve.cosine_search import cosine_top_k
from belgrade_recommender.retrieve.encoder import SentenceEncoder
from belgrade_recommender.retrieve.hard_filters import passes_hard_constraints
from belgrade_recommender.retrieve.service import load_normalized_events
from belgrade_recommender.retrieve.text import build_embedding_text
from belgrade_recommender.schemas.parsed_preferences import ParsedUserPreferences

import run_automated_tests as auto_tests
import run_multilingual_tests as multi_tests
import run_realistic_tests as real_tests
import run_wild_tests as wild_tests

_SEP  = "=" * 72
_THIN = "-" * 72
_GRP  = "~" * 72


def _safe(text: str | None, n: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


def run_member(
    member: dict,
    events: list[NormalizedEvent],
    corpus_matrix: np.ndarray,
    encoder: SentenceEncoder,
) -> tuple[list, dict]:
    prefs: ParsedUserPreferences = member["prefs"]
    query_vec = encoder.encode([member["query"].strip()])[0]
    pool_hits = cosine_top_k(corpus_matrix, query_vec, k=80)

    candidates, attrs_cache = [], {}
    for idx, score in pool_hits:
        ev = events[idx]
        attrs = extract_event_attributes_maybe_lazy_gemini(ev)
        attrs_cache[ev.event_id] = attrs
        if passes_hard_constraints(prefs.hard_constraints, ev, attrs):
            candidates.append((ev, score))
        if len(candidates) >= 20:
            break

    ranked = rank_by_least_misery(candidates, [prefs])[:5]
    return ranked, attrs_cache


def format_member_block(
    member: dict,
    group_id: str,
    ranked: list,
    attrs_cache: dict,
    elapsed: float,
) -> str:
    prefs: ParsedUserPreferences = member["prefs"]
    liked = list(prefs.soft_preferences.liked_types)
    liked_set = {t.lower() for t in liked}

    lines: list[str] = []
    lines.append("")
    lines.append(_SEP)
    lines.append(f"[{group_id}]  {member['name']}")
    lines.append(_THIN)
    lines.append(f"Query      : {_safe(member['query'], 120)}")
    lines.append(f"liked_types: {liked if liked else '(none)'}")
    noise = prefs.soft_preferences.noise_tolerance
    if noise != "unknown":
        lines.append(f"noise_want : {noise}")
    hc = prefs.hard_constraints
    if hc.must_be_free:
        lines.append("must_free  : True")
    if hc.budget_max_rsd:
        lines.append(f"budget     : ≤{hc.budget_max_rsd} RSD")
    if hc.forbidden_event_categories:
        lines.append(f"forbidden  : {hc.forbidden_event_categories}")
    lines.append(f"Time       : {elapsed:.3f}s  |  Results: {len(ranked)}")
    lines.append("")

    if not ranked:
        lines.append("  ⚠  No results — constraints too strict or no coverage.")
        return "\n".join(lines)

    for i, (ev, sem, mn, mean) in enumerate(ranked, 1):
        attrs = attrs_cache.get(ev.event_id)
        raw_tags   = ev.tags or []
        type_hints = list(getattr(attrs, "type_hints", [])) if attrs else []

        # Which liked_types matched (substring, same logic as soft_score)
        th_set = {h.lower() for h in type_hints} | {t.lower() for t in raw_tags}
        matched = [lt for lt in liked if any(lt in x or x in lt for x in th_set)]
        missed  = [lt for lt in liked if lt not in matched]

        lines.append(f"  {i}. {_safe(ev.event_description_resolved, 65)}")
        lines.append(f"     score      : {mn*100:.0f}%  (sem={sem:.3f} min={mn:.3f} mean={mean:.3f})")

        # Raw hashtags from the event (no # prefix stripped for readability)
        if raw_tags:
            tag_str = "  ".join(raw_tags[:15])
            lines.append(f"     raw tags   : {tag_str}")
        else:
            lines.append( "     raw tags   : (none)")

        # Derived type_hints — what the rules engine produced
        if type_hints:
            lines.append(f"     type_hints : {',  '.join(type_hints)}")
        else:
            lines.append( "     type_hints : (none)")

        # noise / outdoor / city from attrs
        if attrs:
            lines.append(
                f"     attrs      : noise={attrs.noise_level_hint}  "
                f"outdoor={attrs.outdoor_hint}  city={attrs.city_hint}  "
                f"free={attrs.price_free_signal}  paid={attrs.price_paid_signal}"
                + (f"  rsd={attrs.price_amount_rsd}" if attrs.price_amount_rsd else "")
                + (f"  eur={attrs.price_amount_eur}" if attrs.price_amount_eur else "")
            )

        # Match analysis
        if liked:
            if matched:
                lines.append(f"     ✓ matched  : {matched}")
            if missed:
                lines.append(f"     ✗ missed   : {missed}")
            if not matched and not missed:
                lines.append( "     (no liked_types to match)")
        else:
            lines.append( "     (user has no liked_types)")

        if ev.link:
            lines.append(f"     link       : {ev.link}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=Path, default=_ROOT / "fixtures" / "events_dataset.jsonl")
    parser.add_argument("--normalized-out", type=Path, default=_ROOT / "data" / "processed" / "events_normalized_test.jsonl")
    parser.add_argument("--skip-ingest",    action="store_true")
    args = parser.parse_args()

    output_path = _ROOT / "run_tag_audit_output.txt"

    # Ingest
    if not args.skip_ingest:
        if not args.dataset.is_file():
            print(f"ERROR: dataset not found: {args.dataset}"); sys.exit(1)
        print(f"Ingesting {args.dataset} …")
        written, skipped = ingest_jsonl_file(args.dataset, args.normalized_out)
        print(f"  {written} written, {skipped} skipped")
    else:
        if not args.normalized_out.is_file():
            print(f"ERROR: normalized file not found: {args.normalized_out}"); sys.exit(1)
        print(f"Skipping ingest, using {args.normalized_out}")

    # Load + encode corpus
    print("Loading corpus …")
    all_events = load_normalized_events(args.normalized_out)
    indexable  = [e for e in all_events if e.include_in_index]
    print(f"  {len(indexable)} indexable events")

    print("Encoding corpus …")
    encoder = SentenceEncoder()
    t0 = time.perf_counter()
    corpus_matrix = encoder.encode([build_embedding_text(e) for e in indexable])
    print(f"  done in {time.perf_counter()-t0:.1f}s — shape {corpus_matrix.shape}")

    # All test suites
    suites: list[tuple[str, list[dict]]] = [
        ("AUTOMATED (G01–G15)",       auto_tests.build_test_groups()),
        ("MULTILINGUAL (R01–R20)",    multi_tests.build_test_groups()),
        ("REALISTIC (L01–L20)",       real_tests.build_test_groups()),
        ("WILD / EXTREME (W01–W21)",  wild_tests.build_test_groups()),
    ]
    total_groups  = sum(len(g) for _, g in suites)
    total_members = sum(len(g["members"]) for _, gs in suites for g in gs)

    out: list[str] = []
    out.append("BELGRADE GROUP RECOMMENDER — TAG AUDIT (all queries)")
    out.append(f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"Suites: {len(suites)}  |  Groups: {total_groups}  |  Members: {total_members}")
    out.append("")
    out.append("Columns per event result:")
    out.append("  raw tags   = hashtags directly on the event (from dataset)")
    out.append("  type_hints = tokens derived by the rules engine from those hashtags + description keywords")
    out.append("  ✓ matched  = user's liked_types that hit at least one type_hint or raw tag (substring)")
    out.append("  ✗ missed   = user's liked_types that matched nothing in this event")
    out.append(_SEP)

    done = errors = 0
    t_wall = time.perf_counter()

    for suite_name, groups in suites:
        out.append(f"\n\n{'#'*72}")
        out.append(f"# SUITE: {suite_name}")
        out.append(f"{'#'*72}")

        for group in groups:
            gid  = group["group_id"]
            desc = group.get("description", "")
            out.append(f"\n\n{_GRP}")
            out.append(f"GROUP : {gid}")
            out.append(f"        {desc}")
            out.append(f"Members : {', '.join(m['name'] for m in group['members'])}")
            out.append(_GRP)

            for member in group["members"]:
                done += 1
                print(f"  [{done:3d}/{total_members}] {gid} / {member['name']}", end="\r")
                t0 = time.perf_counter()
                try:
                    ranked, attrs_cache = run_member(member, indexable, corpus_matrix, encoder)
                    elapsed = time.perf_counter() - t0
                    out.append(format_member_block(member, gid, ranked, attrs_cache, elapsed))
                except Exception as exc:
                    errors += 1
                    elapsed = time.perf_counter() - t0
                    out.append(f"\n{_SEP}")
                    out.append(f"[{gid}] {member['name']}  ERROR ({elapsed:.3f}s): {exc}")

    wall = time.perf_counter() - t_wall
    out.append(f"\n\n{_SEP}")
    out.append("SUMMARY")
    out.append(_SEP)
    out.append(f"Total queries : {done}")
    out.append(f"Errors        : {errors}")
    out.append(f"Wall time     : {wall:.1f}s")
    out.append(_SEP)

    output_path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nDone — {done} queries, {errors} errors, {wall:.1f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

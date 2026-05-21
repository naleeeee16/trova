#!/usr/bin/env python3
"""
Run every member query from all 4 test files through the recommendation pipeline
and save the output in bot-style plain text to run_all_queries_bot_output.txt.

Uses pre-built structured prefs from each test file (skips the OpenAI preference
parsing step to avoid hundreds of extra API calls).  The retrieval + ranking
logic is identical to the live bot.

Usage:
    python run_all_queries_bot_style.py
    python run_all_queries_bot_style.py --skip-ingest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Make test modules importable
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Auto-load .env
# ---------------------------------------------------------------------------
import os
_env_file = _ROOT / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("allqueries")

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

# ---------------------------------------------------------------------------
# Import test group builders
# ---------------------------------------------------------------------------
import run_automated_tests as auto_tests
import run_multilingual_tests as multi_tests
import run_realistic_tests as real_tests
import run_wild_tests as wild_tests

_SEP = "=" * 70
_SEP_THIN = "-" * 70
_SEP_GROUP = "~" * 70


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_member_query(
    member: dict,
    events: list[NormalizedEvent],
    corpus_matrix: np.ndarray,
    encoder: SentenceEncoder,
) -> tuple[list, dict]:
    """Retrieve and rank events for a single member using their pre-built prefs."""
    prefs: ParsedUserPreferences = member["prefs"]
    query = member["query"]

    query_vec = encoder.encode([query.strip()])[0]
    pool_hits = cosine_top_k(corpus_matrix, query_vec, k=80)

    candidates = []
    attrs_cache: dict[str, object] = {}
    for idx, score in pool_hits:
        event = events[idx]
        attrs = extract_event_attributes_maybe_lazy_gemini(event)
        attrs_cache[event.event_id] = attrs
        if passes_hard_constraints(prefs.hard_constraints, event, attrs):
            candidates.append((event, score))
        if len(candidates) >= 20:
            break

    ranked = rank_by_least_misery(candidates, [prefs])[:5]

    mismatches: dict[str, list[str]] = {}
    for ev, _sem, _mn, _mean in ranked:
        attrs = attrs_cache.get(ev.event_id)
        if attrs is None:
            mismatches[ev.event_id] = []
            continue

        issues: list[str] = []

        # Noise mismatch
        noise_hint = getattr(attrs, "noise_level_hint", "unknown")
        noise_pref = prefs.soft_preferences.noise_tolerance
        if noise_hint != "unknown" and noise_pref != "unknown":
            if noise_pref == "low" and noise_hint == "high":
                issues.append("glasno (hoćete tiho)")
            elif noise_pref == "high" and noise_hint == "low":
                issues.append("tiho (hoćete glasno)")

        # Liked-types mismatch
        event_types = set(getattr(attrs, "type_hints", []))
        liked = set(prefs.soft_preferences.liked_types)
        if liked and event_types and not liked.intersection(event_types):
            missing = ", ".join(sorted(liked)[:2])
            issues.append(f"tip ne odgovara ({missing})")

        mismatches[ev.event_id] = issues

    return ranked, mismatches, attrs_cache


# ---------------------------------------------------------------------------
# Formatter (plain text, mirrors bot HTML format)
# ---------------------------------------------------------------------------

def _score_dot(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    if score >= 0.55:
        return "🟡"
    return "🔴"


def _safe(text: str | None, max_len: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:max_len] + ("…" if len(t) > max_len else "")


def format_member_block(
    member: dict,
    group_id: str,
    ranked: list,
    mismatches: dict,
    attrs_cache: dict,
    elapsed: float,
) -> str:
    name = member["name"]
    query = member["query"]
    prefs: ParsedUserPreferences = member["prefs"]

    lines: list[str] = []
    lines.append("")
    lines.append(_SEP)
    lines.append(f"[{group_id}] {name}")
    lines.append(_SEP_THIN)

    # Query + extracted constraints summary
    lines.append(f"Query : {query[:150]}{'…' if len(query) > 150 else ''}")
    hc = prefs.hard_constraints
    sp = prefs.soft_preferences
    pref_parts: list[str] = []
    if sp.liked_types:
        pref_parts.append(f"liked={sp.liked_types}")
    if sp.noise_tolerance != "unknown":
        pref_parts.append(f"noise={sp.noise_tolerance}")
    if hc.must_be_free:
        pref_parts.append("free=True")
    if hc.budget_max_rsd:
        pref_parts.append(f"budget≤{hc.budget_max_rsd}RSD")
    if hc.forbidden_event_categories:
        pref_parts.append(f"forbidden={hc.forbidden_event_categories}")
    if hc.venue_requirements:
        pref_parts.append(f"venue_req={hc.venue_requirements}")
    if hc.dietary_restrictions:
        pref_parts.append(f"dietary={hc.dietary_restrictions}")
    if hc.preferred_days_of_week:
        pref_parts.append(f"days={hc.preferred_days_of_week}")
    if hc.earliest_start_hour_weekday is not None:
        pref_parts.append(f"from={hc.earliest_start_hour_weekday}h")
    if pref_parts:
        lines.append(f"Prefs : {' | '.join(pref_parts)}")
    lines.append(f"Time  : {elapsed:.3f}s | Candidates: {len(ranked)} shown")
    lines.append("")

    if not ranked:
        lines.append("  ⚠️  Nema rezultata — prestrogi uslovi ili nema pokrivenih mesta.")
        return "\n".join(lines)

    lines.append(f"🎯 Top {len(ranked)} mesta za vaše preference:\n")

    for i, (ev, sem, mn, mean) in enumerate(ranked, 1):
        title = _safe(ev.event_description_resolved, 60)
        dot = _score_dot(mn)
        score_pct = f"{mn * 100:.0f}%"

        lines.append(f"{i}. {title}")
        lines.append(f"   {dot} Skor: {score_pct}  (sem={sem:.3f}  min={mn:.3f}  mean={mean:.3f})")

        # Raw hashtags from the dataset
        if ev.tags:
            lines.append(f"   🏷️  Hashtags : {' '.join(ev.tags[:12])}")

        # Derived EventAttributes — what was actually compared
        attrs = attrs_cache.get(ev.event_id)
        if attrs is not None:
            type_str = ", ".join(attrs.type_hints) if attrs.type_hints else "—"
            lines.append(f"   🔎  Types    : {type_str}")
            attr_parts = [
                f"noise={attrs.noise_level_hint}",
                f"outdoor={attrs.outdoor_hint}",
                f"city={attrs.city_hint}",
                f"free={attrs.price_free_signal}",
                f"paid={attrs.price_paid_signal}",
            ]
            if attrs.price_amount_rsd:
                attr_parts.append(f"rsd={attrs.price_amount_rsd}")
            if attrs.price_amount_eur:
                attr_parts.append(f"eur={attrs.price_amount_eur}")
            if attrs.day_of_week_hints:
                attr_parts.append(f"days={attrs.day_of_week_hints}")
            if attrs.date_snippets:
                attr_parts.append(f"dates={attrs.date_snippets[:2]}")
            lines.append(f"   📊  Attrs    : {' | '.join(attr_parts)}")

        gaps = mismatches.get(ev.event_id, [])
        if gaps:
            lines.append(f"   ⚠️  Mismatch  : {', '.join(gaps)}")

        if ev.link:
            lines.append(f"   🔗 {ev.link}")
        else:
            lines.append("   🔗 (no link)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=_ROOT / "fixtures" / "events_dataset.jsonl",
    )
    parser.add_argument(
        "--normalized-out", type=Path,
        default=_ROOT / "data" / "processed" / "events_normalized_test.jsonl",
    )
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    output_path = _ROOT / "run_all_queries_bot_output.txt"

    # Step 1: Ingest
    if not args.skip_ingest:
        if not args.dataset.is_file():
            log.error("Dataset not found: %s", args.dataset)
            sys.exit(1)
        log.info("Step 1/4 — Ingesting %s …", args.dataset)
        t0 = time.perf_counter()
        written, skipped = ingest_jsonl_file(args.dataset, args.normalized_out)
        log.info("Done in %.2fs — %d written, %d skipped", time.perf_counter() - t0, written, skipped)
    else:
        if not args.normalized_out.is_file():
            log.error("Normalized file not found: %s", args.normalized_out)
            sys.exit(1)
        log.info("Step 1/4 — Skipping ingest, using %s", args.normalized_out)

    # Step 2: Load corpus
    log.info("Step 2/4 — Loading corpus …")
    all_events = load_normalized_events(args.normalized_out)
    indexable = [e for e in all_events if e.include_in_index]
    log.info("%d total, %d indexable", len(all_events), len(indexable))
    if not indexable:
        log.error("No indexable events.")
        sys.exit(1)

    # Step 3: Encode corpus once — shared by all queries
    log.info("Step 3/4 — Encoding corpus (one-time) …")
    encoder = SentenceEncoder()
    texts = [build_embedding_text(e) for e in indexable]
    t_enc = time.perf_counter()
    corpus_matrix = encoder.encode(texts)
    log.info(
        "Corpus encoded in %.1fs — shape %s  model=%s",
        time.perf_counter() - t_enc, corpus_matrix.shape, encoder.model_name,
    )

    # Step 4: Collect all groups
    all_suites: list[tuple[str, list[dict]]] = [
        ("AUTOMATED (G01–G15)",        auto_tests.build_test_groups()),
        ("MULTILINGUAL (R01–R20)",     multi_tests.build_test_groups()),
        ("REALISTIC (L01–L20)",        real_tests.build_test_groups()),
        ("WILD / EXTREME (W01–W21)",   wild_tests.build_test_groups()),
    ]

    total_members = sum(
        len(g["members"])
        for _, groups in all_suites
        for g in groups
    )
    total_groups = sum(len(groups) for _, groups in all_suites)
    log.info(
        "Step 4/4 — %d suites, %d groups, %d member queries",
        len(all_suites), total_groups, total_members,
    )

    # ---------------------------------------------------------------------------
    # Build output
    # ---------------------------------------------------------------------------
    out: list[str] = []
    out.append("BELGRADE GROUP RECOMMENDER — ALL MEMBER QUERIES (BOT-STYLE OUTPUT)")
    out.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"Suites    : {len(all_suites)}  |  Groups: {total_groups}  |  Members: {total_members}")
    out.append(_SEP)

    done = 0
    errors = 0
    t_total = time.perf_counter()

    for suite_name, groups in all_suites:
        out.append(f"\n\n{'#' * 70}")
        out.append(f"# TEST SUITE: {suite_name}")
        out.append(f"{'#' * 70}")

        for group in groups:
            group_id = group["group_id"]
            description = group.get("description", "")
            members = group["members"]

            out.append(f"\n\n{_SEP_GROUP}")
            out.append(f"GROUP : {group_id}")
            out.append(f"        {description}")
            out.append(f"Members: {', '.join(m['name'] for m in members)}")
            out.append(_SEP_GROUP)

            for member in members:
                done += 1
                log.info(
                    "  [%d/%d] %s / %s",
                    done, total_members, group_id, member["name"],
                )
                t0 = time.perf_counter()
                try:
                    ranked, mismatches, attrs_cache = run_member_query(
                        member, indexable, corpus_matrix, encoder
                    )
                    elapsed = time.perf_counter() - t0
                    block = format_member_block(
                        member=member,
                        group_id=group_id,
                        ranked=ranked,
                        mismatches=mismatches,
                        attrs_cache=attrs_cache,
                        elapsed=elapsed,
                    )
                    out.append(block)
                    log.info("    → %d results, %.3fs", len(ranked), elapsed)
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    errors += 1
                    log.exception("    FAILED: %s", exc)
                    out.append(f"\n{_SEP}")
                    out.append(f"[{group_id}] {member['name']}")
                    out.append(f"  ERROR ({elapsed:.3f}s): {exc}")

    # Final summary
    wall = time.perf_counter() - t_total
    out.append(f"\n\n{_SEP}")
    out.append("SUMMARY")
    out.append(_SEP)
    out.append(f"Total member queries : {done}")
    out.append(f"Errors               : {errors}")
    out.append(f"Wall time            : {wall:.1f}s")
    out.append(_SEP)

    output_path.write_text("\n".join(out), encoding="utf-8")
    log.info("Output saved to: %s", output_path)
    print(f"\nDone — {done} queries, {errors} errors, {wall:.1f}s total")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

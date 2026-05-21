#!/usr/bin/env python3
"""
Full detailed audit — runs every group from all 4 test suites through the
group recommender pipeline and writes a rich per-event breakdown:

  • member queries + preferences
  • per-event: description, link, score (sem/min/mean)
  • per-event: raw tags, type_hints, attrs (noise/outdoor/city/free/paid/rsd/eur)
  • per-member: ✓ matched liked_types  /  ✗ missed liked_types

Output is written to run_full_detailed_audit_output.txt

Usage:
    python run_full_detailed_audit.py
    python run_full_detailed_audit.py --skip-ingest
    python run_full_detailed_audit.py --top 5        # show top-N events per group (default 10)
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime
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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(levelname)-8s  %(message)s")

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
from belgrade_recommender.schemas.parsed_preferences import ParsedUserPreferences

from belgrade_recommender.bot.pipeline import _is_non_event

import run_automated_tests   as auto_tests
import run_multilingual_tests as multi_tests
import run_realistic_tests    as real_tests
import run_wild_tests         as wild_tests

_SEP  = "=" * 80
_THIN = "-" * 80
_GRP  = "~" * 80


def _safe(text: str | None, n: int) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


# Same expansion map as the live bot pipeline — niche cultural terms that
# the multilingual model hasn't seen enough to embed well on their own.
_EMBED_EXPANSION: dict[str, str] = {
    "turbofolk":        "turbofolk Serbian folk pop music party dance",
    "folk_traditional": "traditional folk music Serbian narodni kafana",
    "kafana":           "kafana Serbian tavern folk music traditional drinks",
    "rakija":           "rakija Serbian traditional spirits folk tavern",
    "splav":            "splav river raft party club Belgrade nightlife",
    "dark_techno":      "dark techno industrial underground club night",
    "dark_ambient":     "dark ambient industrial gothic experimental music",
}


def run_group(
    group: dict,
    events: list[NormalizedEvent],
    corpus_matrix: np.ndarray,
    encoder: SentenceEncoder,
    top_n: int,
) -> tuple[list, dict, float]:
    members = group["members"]
    hards   = [m["prefs"].hard_constraints for m in members]
    prefs   = [m["prefs"] for m in members]
    # Pre-embed each member's liked_types once — used for cosine-based scoring
    # (same path as the live bot, not the substring fallback)
    liked_vecs_list: list = []
    for m in members:
        lt_terms = m["prefs"].soft_preferences.liked_types
        if lt_terms:
            phrases = [_EMBED_EXPANSION.get(t, t.replace("_", " ")) for t in lt_terms]
            liked_vecs_list.append(encoder.encode(phrases))
        else:
            liked_vecs_list.append(None)

    # Encode each member's query independently then average — same as live bot.
    # Concatenating all text into one string lets longer queries dominate; averaging
    # gives each member equal weight in the semantic search regardless of query length.
    t0 = time.perf_counter()
    per_user_vecs = []
    for mi, m in enumerate(members):
        lt_terms = m["prefs"].soft_preferences.liked_types or []
        liked_str = " ".join(_EMBED_EXPANSION.get(t, t.replace("_", " ")) for t in lt_terms)
        per_query = (m["query"] + ("\n\n" + liked_str if liked_str else "")).strip()
        per_user_vecs.append(encoder.encode([per_query])[0])
    query_vec = np.mean(per_user_vecs, axis=0)
    pool_hits = cosine_top_k(corpus_matrix, query_vec, k=80)

    attrs_cache: dict = {}
    event_vecs:  dict = {}
    candidates: list[tuple[NormalizedEvent, float]] = []
    for idx, score in pool_hits:
        ev = events[idx]
        if _is_non_event(ev.event_description_resolved, ev.tags):
            continue
        attrs = extract_event_attributes_maybe_lazy_gemini(ev)
        attrs_cache[ev.event_id] = attrs
        if passes_all_hard_constraints(hards, ev, attrs):
            candidates.append((ev, score))
            event_vecs[ev.event_id] = corpus_matrix[idx]
        if len(candidates) >= 20:
            break

    ranked = rank_by_least_misery(
        candidates, prefs,
        liked_vecs_list=liked_vecs_list,
        event_vecs=event_vecs,
    )[:top_n]
    latency = time.perf_counter() - t0
    return ranked, attrs_cache, latency, len(candidates), liked_vecs_list, event_vecs


def format_group(
    group: dict,
    suite_name: str,
    ranked: list,
    attrs_cache: dict,
    latency: float,
    n_candidates: int,
    top_n: int,
    liked_vecs_list: list | None = None,
    event_vecs: dict | None = None,
) -> str:
    members = group["members"]
    gid     = group["group_id"]
    desc    = group.get("description", "")

    lines: list[str] = []

    # ── Group header ────────────────────────────────────────────────────────
    lines += [
        "",
        _GRP,
        f"GROUP   : {gid}",
        f"SUITE   : {suite_name}",
        f"DESC    : {desc}",
        f"MEMBERS : {', '.join(m['name'] for m in members)}",
        _THIN,
    ]

    # ── Per-member preference summary ────────────────────────────────────────
    for m in members:
        p: ParsedUserPreferences = m["prefs"]
        hc = p.hard_constraints
        sp = p.soft_preferences
        lines.append(f"  [{m['name']}]")
        lines.append(f"    query      : {_safe(m['query'], 110)}")
        lines.append(f"    liked_types: {list(sp.liked_types) if sp.liked_types else '(none)'}")
        lines.append(f"    noise      : {sp.noise_tolerance}")
        constraints = []
        if hc.must_be_free:
            constraints.append("must_be_free=True")
        if hc.budget_max_rsd:
            constraints.append(f"budget≤{hc.budget_max_rsd}RSD")
        if hc.forbidden_event_categories:
            constraints.append(f"forbidden={hc.forbidden_event_categories}")
        if hc.earliest_start_hour_weekday:
            constraints.append(f"earliest={hc.earliest_start_hour_weekday}h")
        lines.append(f"    hard       : {', '.join(constraints) if constraints else '(none)'}")
    lines.append(_THIN)

    # ── Stats ────────────────────────────────────────────────────────────────
    lines.append(f"  Latency    : {latency:.3f}s")
    lines.append(f"  Candidates : {n_candidates}  (after semantic + hard filter)")
    lines.append(f"  Ranked     : {len(ranked)}")
    if not ranked:
        lines.append("")
        lines.append("  ⚠  WARNING: No results returned — constraints too strict or no coverage.")
        lines.append(_GRP)
        return "\n".join(lines)

    lines.append("")

    # ── Ranked events ─────────────────────────────────────────────────────
    for i, (ev, sem, mn, mean) in enumerate(ranked, 1):
        attrs    = attrs_cache.get(ev.event_id)
        raw_tags = ev.tags or []
        type_hints = list(getattr(attrs, "type_hints", [])) if attrs else []

        lines.append(f"  ── #{i} {'─'*60}")
        lines.append(f"  Title : {_safe(ev.event_description_resolved, 100)}")
        if ev.link:
            lines.append(f"  Link  : {ev.link}")
        lines.append(f"  Score : sem={sem:.4f}  min_soft={mn:.4f}  mean_soft={mean:.4f}  [{mn*100:.0f}% group score]")
        lines.append("")

        # tags & derived hints
        if raw_tags:
            lines.append(f"  raw tags   : {' | '.join(raw_tags)}")
        else:
            lines.append("  raw tags   : (none)")

        if type_hints:
            lines.append(f"  type_hints : {' | '.join(type_hints)}")
        else:
            lines.append("  type_hints : (none)")

        # attrs
        if attrs:
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
            lines.append(f"  attrs      : {' | '.join(attr_parts)}")
        lines.append("")

        # per-member match analysis — cosine path (same as live bot)
        ev_vec = event_vecs.get(ev.event_id) if event_vecs else None
        for mi, m in enumerate(members):
            p: ParsedUserPreferences = m["prefs"]
            liked = list(p.soft_preferences.liked_types)
            if not liked:
                lines.append(f"  [{m['name']}]  (no liked_types)")
                continue
            lv = liked_vecs_list[mi] if liked_vecs_list else None
            if lv is not None and ev_vec is not None:
                ev_norm = float(np.linalg.norm(ev_vec)) + 1e-9
                lt_norms = np.linalg.norm(lv, axis=1) + 1e-9
                cosines = (lv @ ev_vec) / (lt_norms * ev_norm)
                hits   = [(lt, float(c)) for lt, c in zip(liked, cosines) if c >= 0.35]
                misses = [(lt, float(c)) for lt, c in zip(liked, cosines) if c < 0.35]
                hit_str  = f"✓ {[f'{lt}({c:.2f})' for lt,c in hits]}"  if hits  else "✓ (none)"
                miss_str = f"✗ {[f'{lt}({c:.2f})' for lt,c in misses]}" if misses else "✗ (none)"
            else:
                th_set = {h.lower() for h in type_hints} | {t.lower() for t in raw_tags}
                matched = [lt for lt in liked if any(lt in x or x in lt for x in th_set)]
                missed  = [lt for lt in liked if lt not in matched]
                hit_str  = f"✓ {matched}" if matched else "✓ (none)"
                miss_str = f"✗ {missed}"  if missed  else "✗ (none)"
            lines.append(f"  [{m['name']}]  {hit_str}    {miss_str}")

        lines.append("")

    lines.append(_GRP)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=Path, default=_ROOT / "fixtures" / "events_dataset.jsonl")
    parser.add_argument("--normalized-out", type=Path, default=_ROOT / "data" / "processed" / "events_normalized_test.jsonl")
    parser.add_argument("--skip-ingest",    action="store_true")
    parser.add_argument("--top",            type=int, default=10, help="Top-N events per group (default 10)")
    args = parser.parse_args()

    output_path = _ROOT / "run_full_detailed_audit_output.txt"

    # ── Ingest ────────────────────────────────────────────────────────────
    if not args.skip_ingest:
        if not args.dataset.is_file():
            print(f"ERROR: dataset not found: {args.dataset}"); sys.exit(1)
        print(f"Ingesting {args.dataset} …")
        written, skipped = ingest_jsonl_file(args.dataset, args.normalized_out)
        print(f"  {written} written, {skipped} skipped")
    else:
        if not args.normalized_out.is_file():
            print(f"ERROR: normalized file not found: {args.normalized_out}"); sys.exit(1)

    # ── Load + encode corpus ──────────────────────────────────────────────
    print("Loading corpus …")
    all_events = load_normalized_events(args.normalized_out)
    indexable  = [e for e in all_events if e.include_in_index]
    print(f"  {len(indexable)} indexable events")

    print("Encoding corpus …")
    encoder = SentenceEncoder()
    t0 = time.perf_counter()
    corpus_matrix = encoder.encode([build_embedding_text(e) for e in indexable])
    print(f"  done in {time.perf_counter()-t0:.1f}s — shape {corpus_matrix.shape}")

    # ── All suites ────────────────────────────────────────────────────────
    suites: list[tuple[str, list[dict]]] = [
        ("AUTOMATED (G01–G15)",       auto_tests.build_test_groups()),
        ("MULTILINGUAL (R01–R20)",    multi_tests.build_test_groups()),
        ("WILD / EXTREME (W01–W21)",  wild_tests.build_test_groups()),
        ("REALISTIC (L01–L20)",       real_tests.build_test_groups()),
    ]
    total_groups  = sum(len(gs) for _, gs in suites)
    total_members = sum(len(g["members"]) for _, gs in suites for g in gs)

    # ── Output header ────────────────────────────────────────────────────
    out: list[str] = []
    out.append(_SEP)
    out.append("BELGRADE GROUP RECOMMENDER — FULL DETAILED AUDIT")
    out.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"Suites    : {len(suites)}  |  Groups: {total_groups}  |  Members: {total_members}")
    out.append(f"Top-N     : {args.top} events per group")
    out.append(_SEP)
    out.append("")
    out.append("Per-event columns:")
    out.append("  raw tags   = hashtags directly on the event (from dataset)")
    out.append("  type_hints = tokens derived by the rules engine")
    out.append("  attrs      = noise / outdoor / city / free / paid / price")
    out.append("  [Member]   = ✓ matched liked_types  |  ✗ missed liked_types")
    out.append(_SEP)

    done = errors = zero_groups = 0
    t_wall = time.perf_counter()

    for suite_name, groups in suites:
        out.append(f"\n\n{'#'*80}")
        out.append(f"# SUITE: {suite_name}")
        out.append(f"{'#'*80}")

        for group in groups:
            done += 1
            print(f"  [{done:3d}/{total_groups}] {group['group_id']}", end="\r")
            try:
                ranked, attrs_cache, latency, n_cand, liked_vecs_list, event_vecs = run_group(
                    group, indexable, corpus_matrix, encoder, args.top
                )
                if not ranked:
                    zero_groups += 1
                block = format_group(
                    group, suite_name, ranked, attrs_cache, latency, n_cand, args.top,
                    liked_vecs_list=liked_vecs_list,
                    event_vecs=event_vecs,
                )
                out.append(block)
            except Exception as exc:
                errors += 1
                out.append(f"\n{_SEP}")
                out.append(f"ERROR in {group['group_id']}: {exc}")

    wall = time.perf_counter() - t_wall

    # ── Summary ───────────────────────────────────────────────────────────
    out.append(f"\n\n{_SEP}")
    out.append("SUMMARY")
    out.append(_SEP)
    out.append(f"Total groups   : {done}")
    out.append(f"Zero-result    : {zero_groups}")
    out.append(f"Errors         : {errors}")
    out.append(f"Wall time      : {wall:.1f}s")
    out.append(_SEP)

    output_path.write_text("\n".join(out), encoding="utf-8")
    print(f"\nDone — {done} groups, {zero_groups} zero-result, {errors} errors, {wall:.1f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

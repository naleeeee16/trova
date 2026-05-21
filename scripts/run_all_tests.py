#!/usr/bin/env python3
"""
Master test runner — runs all four Belgrade Group Recommender test suites and
writes the full output + a summary table to a timestamped .txt file.

Usage:
    python run_all_tests.py                      # full ingest + all suites
    python run_all_tests.py --skip-ingest        # reuse existing normalized data
    python run_all_tests.py --output results.txt # custom output file name
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent

SUITES: list[tuple[str, str]] = [
    ("AUTOMATED  (SR/EN baseline)",       "run_automated_tests.py"),
    ("MULTILINGUAL  (RU+SR+EN)",          "run_multilingual_tests.py"),
    ("WILD / EXTREME  (stress test)",     "run_wild_tests.py"),
    ("REALISTIC  (everyday scenarios)",   "run_realistic_tests.py"),
    ("G02 TIJANA+FILIP  (tag audit)",     "run_g02_tijana_filip_audit.py"),
]

_SEP  = "=" * 72
_THIN = "-" * 72


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_suite(script: str, *, skip_ingest: bool) -> tuple[str, int, float]:
    """Spawn script as a subprocess; return (combined_output, exit_code, seconds)."""
    cmd = [sys.executable, str(_ROOT / script)]
    if skip_ingest:
        cmd.append("--skip-ingest")

    # Force UTF-8 stdout in the child process so Serbian/Cyrillic characters don't crash
    import os as _os
    env = _os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout so order is preserved
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=_ROOT,
    )
    elapsed = time.perf_counter() - t0
    return proc.stdout or "", proc.returncode, elapsed


# ---------------------------------------------------------------------------
# Metrics parser
# ---------------------------------------------------------------------------

def parse_metrics(output: str) -> dict:
    """Pull key numbers out of a suite's stdout."""
    m = {
        "groups_run": 0,
        "groups_zero": 0,
        "expected_zero_pass": 0,
        "hard_ok": 0,
        "hard_warn": 0,
        "candidate_counts": [],
        "latencies": [],
        "ranked_counts": [],
    }

    for line in output.splitlines():
        s = line.strip()

        cand = re.search(r"Candidates\s*:\s*(\d+)", s)
        if cand:
            m["groups_run"] += 1
            m["candidate_counts"].append(int(cand.group(1)))

        ranked = re.search(r"^Ranked\s*:\s*(\d+)", s)
        if ranked:
            m["ranked_counts"].append(int(ranked.group(1)))

        lat = re.search(r"^Latency\s*:\s*([\d.]+)s", s)
        if lat:
            m["latencies"].append(float(lat.group(1)))

        if "WARNING: No results returned" in s:
            m["groups_zero"] += 1

        if "PASS" in s and "Correctly returned 0 results" in s:
            m["expected_zero_pass"] += 1

        if "Hard constraints" in s:
            if "[OK]" in s:
                m["hard_ok"] += 1
            elif "[WARN]" in s:
                m["hard_warn"] += 1

    return m


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(
    suite_results: list[tuple[str, str, dict, int, float]],
    total_elapsed: float,
    timestamp: str,
) -> str:
    lines: list[str] = []

    lines += [
        _SEP,
        "  MASTER TEST RUN — FINAL SUMMARY",
        f"  Generated : {timestamp}",
        f"  Wall time : {total_elapsed:.1f}s",
        _SEP,
        "",
    ]

    # Per-suite table header
    col = f"  {'Suite':<36} {'Groups':>6} {'HasRes':>6} {'ZeroR':>6} {'AvgCnd':>7} {'AvgLat':>8} {'HardOK':>7} {'Status':>7}"
    lines.append(col)
    lines.append("  " + "-" * 70)

    grand_groups = grand_with = grand_zero = grand_exp_zero = 0
    grand_hard_ok = grand_hard_warn = 0
    grand_cands: list[int] = []
    grand_lats:  list[float] = []
    grand_ranked: list[int] = []

    suite_rows: list[str] = []
    for label, script, m, code, elapsed in suite_results:
        n = m["groups_run"]
        zero = m["groups_zero"]
        with_res = n - zero
        avg_cands = sum(m["candidate_counts"]) / n if n else 0.0
        avg_lat   = sum(m["latencies"]) / len(m["latencies"]) if m["latencies"] else 0.0
        status    = "PASS" if code == 0 else "FAIL"

        grand_groups    += n
        grand_with      += with_res
        grand_zero      += zero
        grand_exp_zero  += m["expected_zero_pass"]
        grand_hard_ok   += m["hard_ok"]
        grand_hard_warn += m["hard_warn"]
        grand_cands     += m["candidate_counts"]
        grand_lats      += m["latencies"]
        grand_ranked    += m["ranked_counts"]

        suite_rows.append(
            f"  {label:<36} {n:>6} {with_res:>6} {zero:>6} "
            f"{avg_cands:>7.1f} {avg_lat:>7.3f}s {m['hard_ok']:>7} {status:>7}"
        )

    lines += suite_rows
    lines.append("  " + "-" * 70)

    # Grand totals row
    g_avg_cands = sum(grand_cands) / grand_groups if grand_groups else 0.0
    g_avg_lat   = sum(grand_lats)  / len(grand_lats) if grand_lats else 0.0
    lines.append(
        f"  {'TOTAL / AVERAGE':<36} {grand_groups:>6} {grand_with:>6} {grand_zero:>6} "
        f"{g_avg_cands:>7.1f} {g_avg_lat:>7.3f}s {grand_hard_ok:>7}"
    )

    # Key metrics block
    pct_with = 100 * grand_with  / grand_groups if grand_groups else 0.0
    pct_hard = 100 * grand_hard_ok / (grand_hard_ok + grand_hard_warn) if (grand_hard_ok + grand_hard_warn) else 0.0
    full_10  = sum(1 for r in grand_ranked if r == 10)

    lines += [
        "",
        _SEP,
        "  KEY METRICS",
        _THIN,
        f"  Total groups tested          : {grand_groups}",
        f"  Groups returning results     : {grand_with}  ({pct_with:.0f}%)",
        f"  Groups returning 0 results   : {grand_zero}",
        f"  Expected-zero groups OK      : {grand_exp_zero}",
        f"  Groups returning full top-10 : {full_10}",
        f"  Hard constraint pass rate    : {grand_hard_ok}/{grand_hard_ok+grand_hard_warn} ({pct_hard:.0f}%)",
        f"  Hard constraint warnings     : {grand_hard_warn}",
        f"  Avg candidates per group     : {g_avg_cands:.1f}",
        f"  Avg query latency            : {g_avg_lat:.3f}s",
        f"  Total wall time              : {total_elapsed:.1f}s",
        _SEP,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all test suites and write results to a .txt file.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion (reuse normalized file).")
    parser.add_argument("--output", type=Path, default=None, help="Output .txt path (default: auto-timestamped).")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_file: Path = args.output or (_ROOT / f"test_results_{timestamp}.txt")

    print(_SEP)
    print("  Belgrade Group Recommender — Master Test Runner")
    print(_SEP)
    print(f"  Suites  : {len(SUITES)}")
    print(f"  Mode    : {'--skip-ingest' if args.skip_ingest else 'full ingest on first suite'}")
    print(f"  Output  : {out_file}")
    print(_SEP)
    print()

    parts: list[str] = []
    suite_results: list[tuple[str, str, dict, int, float]] = []

    # File header
    parts.append(
        f"BELGRADE GROUP RECOMMENDER — FULL TEST SUITE RUN\n"
        f"Timestamp : {timestamp}\n"
        f"Mode      : {'skip-ingest' if args.skip_ingest else 'full ingest'}\n"
        f"{_SEP}\n"
    )

    t_total = time.perf_counter()

    for idx, (label, script) in enumerate(SUITES, 1):
        # Only first suite does ingest (writes the normalized file); all others reuse it
        skip = args.skip_ingest or (idx > 1)

        print(f"[{idx}/{len(SUITES)}]  {label}")
        print(f"         Running {script} {'--skip-ingest' if skip else '(with ingest)'} ...")

        suite_header = (
            f"\n{_SEP}\n"
            f"  SUITE {idx}/{len(SUITES)}: {label}\n"
            f"  Script : {script}\n"
            f"{_SEP}\n"
        )
        parts.append(suite_header)

        output, code, elapsed = run_suite(script, skip_ingest=skip)

        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"         Finished in {elapsed:.1f}s — {status}\n")

        parts.append(output)
        parts.append(f"\n[Suite finished in {elapsed:.1f}s — exit code: {code}]\n")

        m = parse_metrics(output)
        suite_results.append((label, script, m, code, elapsed))

    total_elapsed = time.perf_counter() - t_total

    # Build summary
    summary = build_summary(suite_results, total_elapsed, timestamp)
    parts.append(f"\n\n{summary}\n")

    # Write file
    full_text = "".join(parts)
    out_file.write_text(full_text, encoding="utf-8")

    # Print summary to console too
    print()
    print(summary)
    print()
    print(f"Full output written to: {out_file}")


if __name__ == "__main__":
    main()

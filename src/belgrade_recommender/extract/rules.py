"""Regex and keyword helpers for rule-based event attribute extraction."""

from __future__ import annotations

import re
from typing import Literal

CityHint = Literal["belgrade", "novi_sad", "serbia_other", "unknown"]

_FREE = re.compile(
    r"\b("
    r"free\s+entry|free\s+admission|no\s+cover|"
    r"besplat(?:an|ni|an)?|ulaz\s+je\s+slobodan|"
    r"вход\s+свободн|бесплатн"
    r")\b",
    re.IGNORECASE,
)

_RSD = re.compile(r"(\d[\d.,]*)\s*(?:RSD|дин\.?|din\.?)\b", re.IGNORECASE)
_EUR = re.compile(r"(\d[\d.,]*)\s*(?:EUR|€|euros?)\b", re.IGNORECASE)

_DATE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}[./-]20\d{2}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+20\d{2}\b",
    re.IGNORECASE,
)


def detect_free_entry(text: str) -> bool:
    return bool(_FREE.search(text or ""))


def _parse_amount(raw: str) -> int | None:
    s = raw.replace(" ", "").replace(",", "")
    if "." in s and s.rindex(".") > 0:
        parts = s.split(".")
        if len(parts[-1]) == 3 and parts[0].isdigit():
            s = "".join(parts)
        elif len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            s = parts[0] + parts[1]
        else:
            s = s.replace(".", "")
    if not s.isdigit():
        return None
    v = int(s)
    return v if v >= 0 else None


def max_rsd_amount(text: str) -> int | None:
    best: int | None = None
    for m in _RSD.finditer(text or ""):
        v = _parse_amount(m.group(1))
        if v is None:
            continue
        best = v if best is None else max(best, v)
    return best


def max_eur_amount(text: str) -> int | None:
    best: int | None = None
    for m in _EUR.finditer(text or ""):
        v = _parse_amount(m.group(1))
        if v is None:
            continue
        best = v if best is None else max(best, v)
    return best


def extract_date_snippets(text: str, limit: int = 5) -> list[str]:
    out: list[str] = []
    for m in _DATE.finditer(text or ""):
        s = m.group(0).strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def detect_city_hint(text: str) -> CityHint:
    t = (text or "").lower()
    if "novi sad" in t:
        return "novi_sad"
    if "belgrade" in t or "beograd" in t or "белград" in (text or "").lower():
        return "belgrade"
    if "serbia" in t or "srbij" in t or "серб" in (text or "").lower():
        return "serbia_other"
    return "unknown"


_DAY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # English
    (re.compile(r"\bmonday\b", re.I), "monday"),
    (re.compile(r"\btuesday\b", re.I), "tuesday"),
    (re.compile(r"\bwednesday\b", re.I), "wednesday"),
    (re.compile(r"\bthursday\b", re.I), "thursday"),
    (re.compile(r"\bfriday\b", re.I), "friday"),
    (re.compile(r"\bsaturday\b", re.I), "saturday"),
    (re.compile(r"\bsunday\b", re.I), "sunday"),
    # Serbian Latin (full forms only to avoid false positives)
    (re.compile(r"\bponedelj[ae]k\b", re.I), "monday"),
    (re.compile(r"\butorak\b", re.I), "tuesday"),
    (re.compile(r"\bsrij?eda\b|\bsreda\b", re.I), "wednesday"),
    (re.compile(r"\b[čc]etvrt[ae]k\b", re.I | re.UNICODE), "thursday"),
    (re.compile(r"\bpetak\b", re.I), "friday"),
    (re.compile(r"\bsubota\b", re.I), "saturday"),
    (re.compile(r"\bnedjelj[ae]\b|\bnedelj[ae]\b", re.I), "sunday"),
    # Russian Cyrillic (nom + common inflected forms)
    (re.compile(r"\bпонедельник[а-я]*\b", re.I), "monday"),
    (re.compile(r"\bвторник[а-я]*\b", re.I), "tuesday"),
    (re.compile(r"\bсред[ауы]\b|\bсреда\b", re.I), "wednesday"),
    (re.compile(r"\bчетверг[а-я]*\b", re.I), "thursday"),
    (re.compile(r"\bпятниц[аыу]\b|\bпятница\b", re.I), "friday"),
    (re.compile(r"\bсуббот[аыу]\b|\bсуббота\b", re.I), "saturday"),
    (re.compile(r"\bвоскресень[яе]\b|\bвоскресенье\b", re.I), "sunday"),
]

_DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def extract_day_of_week(text: str) -> list[str]:
    """Return canonical English day names found in text (SR/RU/EN), deduped, Mon→Sun order."""
    found: set[str] = set()
    t = text or ""
    for pattern, canonical in _DAY_PATTERNS:
        if pattern.search(t):
            found.add(canonical)
    return [d for d in _DAY_ORDER if d in found]


def paid_language_hint(text: str) -> bool:
    """Heuristic: explicit paid / ticket wording without being only 'free'."""

    t = text or ""
    return bool(
        re.search(r"\b(tickets?\s+from|ticket\s+price|from\s+\d+\s*(?:RSD|EUR|€|din))\b", t, re.I)
        or re.search(r"\b\d+\s*(?:RSD|EUR|€)\b", t)
    )

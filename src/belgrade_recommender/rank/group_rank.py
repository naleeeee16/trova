"""Group ranking: per-user soft scores + least-misery style aggregation"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Month names used for dedup key normalization.
# We truncate the text AT the first month name (within the first 80 chars) so that
# "Free concerts in Belgrade in June. As always…" and "…in May  I don't know…"
# produce the same key — the text after the month diverges, so removing isn't enough.
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|januar|februar|mart|april|maj|jun|jul|avgust|septembar|oktobar|novembar|decembar)\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b\d{4}\b")

from belgrade_recommender.extract.attributes import extract_event_attributes_maybe_lazy_gemini
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.rank.soft_score import soft_adjusted_score
from belgrade_recommender.schemas.parsed_preferences import (
    ParsedHardConstraints,
    ParsedSoftPreferences,
    ParsedUserPreferences,
)


def per_user_soft_scores(
    members: Sequence[ParsedUserPreferences],
    event: NormalizedEvent,
    semantic_similarity: float,
    liked_vecs_list: "list | None" = None,
    event_vec: "object | None" = None,
) -> list[float]:
    attrs = extract_event_attributes_maybe_lazy_gemini(event)
    scores = []
    for i, p in enumerate(members):
        lv = liked_vecs_list[i] if liked_vecs_list is not None else None
        scores.append(soft_adjusted_score(p, event, attrs, semantic_similarity, lv, event_vec))
    return scores


def _dedup_key(event: NormalizedEvent) -> str:
    """Collapse recurring monthly series to one dedup key.

    Strategy: truncate the text at the first month name found within the first
    80 characters (template titles differ only after the month word), then strip
    4-digit years, normalize punctuation, and take 60 chars.
    """
    text = event.event_description_resolved.lower().strip()
    m = _MONTH_RE.search(text[:80])
    if m:
        text = text[: m.start()]
    text = _YEAR_RE.sub("", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:60]


def rank_by_least_misery(
    candidates: Sequence[tuple[NormalizedEvent, float]],
    members: Sequence[ParsedUserPreferences],
    liked_vecs_list: "list | None" = None,
    event_vecs: "dict | None" = None,
) -> list[tuple[NormalizedEvent, float, float, float]]:
    """
    Sort candidates by (min user score, then mean user score) descending.
    Deduplicates results from the same recurring event series before returning.

    Args:
        liked_vecs_list: list of np.ndarray (one per member) with pre-embedded liked_types.
        event_vecs: dict mapping event_id → np.ndarray with the event's corpus embedding.

    Returns tuples ``(event, semantic_sim, min_soft, mean_soft)``.
    """

    rows: list[tuple[NormalizedEvent, float, float, float]] = []
    for event, sem in candidates:
        ev_vec = event_vecs.get(event.event_id) if event_vecs else None
        scores = per_user_soft_scores(members, event, sem, liked_vecs_list, ev_vec)
        if not scores:
            continue
        mi = min(scores)
        avg = sum(scores) / len(scores)
        rows.append((event, sem, mi, avg))
    rows.sort(key=lambda r: (r[2], r[3]), reverse=True)

    seen: set[str] = set()
    deduped: list[tuple[NormalizedEvent, float, float, float]] = []
    for row in rows:
        key = _dedup_key(row[0])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def fixture_user_to_prefs(user: dict) -> ParsedUserPreferences:
    """Map one ``synthetic_users.json`` user object to ``ParsedUserPreferences``."""

    return ParsedUserPreferences(
        summary_one_line="",
        hard_constraints=ParsedHardConstraints.model_validate(user["hard_constraints"]),
        soft_preferences=ParsedSoftPreferences.model_validate(user["soft_preferences"]),
    )


def combined_group_query_text(users: Sequence[dict]) -> str:
    return "\n\n".join(u["preferences_plain_text"].strip() for u in users if u.get("preferences_plain_text"))

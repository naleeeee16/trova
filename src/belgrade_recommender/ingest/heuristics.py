from __future__ import annotations

"""Phase 1b: coarse rules for which rows are worth indexing for retrieval."""

MIN_DESCRIPTION_LEN = 80

# Lowercase substrings in description or tags, likely not a single actionable event post.
EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "good morning",
    "dobro jutro",
    "ingredients:",
    "preheat the oven",
    "number of the day",
    "main news",
    "главные новости",
    "events for the week",
    "события и тусовки на неделе",
    "events* in the week",
)


def classify_for_retrieval_index(
    *,
    event_description_resolved: str,
    tags: list[str],
    event_description_raw: str,
) -> tuple[bool, str | None]:
    """
    Return (include_in_index, exclude_reason).
    Reasons are short machine codes for debugging and reports, not user-facing prose.
    """

    text = (event_description_resolved or "").strip()
    lower = text.lower()
    tags_lower = [t.lower() for t in tags]
    combined = lower + " " + " ".join(tags_lower)
    raw_lower = (event_description_raw or "").lower()

    if len(text) < MIN_DESCRIPTION_LEN:
        return False, "too_short"

    for phrase in EXCLUDE_SUBSTRINGS:
        if phrase in combined or phrase in raw_lower:
            return False, f"keyword:{phrase[:24].replace(' ', '_')}"

    if "promo" in tags_lower:
        return False, "tag:promo"

    weekday_hits = sum(
        1
        for w in (
            "monday,",
            "tuesday,",
            "wednesday,",
            "thursday,",
            "friday,",
            "saturday,",
            "sunday,",
        )
        if w in lower
    )
    if weekday_hits >= 3:
        return False, "digest_week_pattern"

    return True, None

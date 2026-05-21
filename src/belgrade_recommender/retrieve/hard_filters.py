"""Hard constraints vs structured event signals

Soft preferences belong in ranking later, this module only removes or keeps
candidates for hard rules aligned with ``EventAttributes`` + tags.
"""

from __future__ import annotations

from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.schemas.event_attributes import EventAttributes
from collections.abc import Sequence

from belgrade_recommender.schemas.parsed_preferences import ParsedHardConstraints


def passes_hard_constraints(
    hard: ParsedHardConstraints,
    event: NormalizedEvent,
    attrs: EventAttributes,
) -> bool:
    """Return False if the event clearly violates a hard constraint."""

    if not _city_ok(hard.city, attrs.city_hint):
        return False
    if not _free_and_budget_ok(hard, attrs):
        return False
    if not _forbidden_categories_ok(hard, event, attrs):
        return False
    if not _adults_only_ok(hard, attrs):
        return False
    if not _day_of_week_ok(hard, attrs):
        return False
    if not _venue_requirements_ok(hard, event, attrs):
        return False
    if not _dietary_restrictions_ok(hard, event, attrs):
        return False
    return True


def passes_all_hard_constraints(
    hards: Sequence[ParsedHardConstraints],
    event: NormalizedEvent,
    attrs: EventAttributes,
) -> bool:
    """Group rule: event must satisfy every member's hard constraints (intersection)."""

    return all(passes_hard_constraints(h, event, attrs) for h in hards)


def _city_ok(city_pref: str | None, hint: str) -> bool:
    if not city_pref or city_pref == "unknown":
        return True
    if city_pref == "belgrade":
        return hint not in ("novi_sad", "serbia_other")
    if city_pref in ("belgrade_or_daytrip_serbia", "serbia_other"):
        return True
    return True


def _has_paid_signal(attrs: EventAttributes) -> bool:
    return bool(attrs.price_paid_signal or attrs.price_amount_rsd or attrs.price_amount_eur)


def _exceeds_budget(attrs: EventAttributes, cap: int) -> bool:
    if attrs.price_amount_rsd is not None and attrs.price_amount_rsd > cap:
        return True
    if attrs.price_amount_eur is not None and attrs.price_amount_eur > 0:
        return attrs.price_amount_eur * 120 > cap
    return False


def _free_and_budget_ok(hard: ParsedHardConstraints, attrs: EventAttributes) -> bool:
    if hard.must_be_free is True:
        # PASS: explicit free signal, or no price info (benefit of the doubt).
        # FAIL: any explicit paid signal.
        return attrs.price_free_signal or not _has_paid_signal(attrs)

    if hard.must_be_free_or_under_budget is True and hard.budget_max_rsd is not None:
        return not _exceeds_budget(attrs, hard.budget_max_rsd)

    # budget_max_rsd alone (without must_be_free_or_under_budget) must also be enforced
    if hard.budget_max_rsd is not None:
        return not _exceeds_budget(attrs, hard.budget_max_rsd)

    return True


def _day_of_week_ok(hard: ParsedHardConstraints, attrs: EventAttributes) -> bool:
    if not hard.preferred_days_of_week:
        return True
    # Event day unknown — be permissive; we don't want to over-filter undated events
    if not attrs.day_of_week_hints:
        return True
    preferred = {d.lower() for d in hard.preferred_days_of_week}
    hints = {d.lower() for d in attrs.day_of_week_hints}
    return bool(preferred & hints)


_ADULTS_ONLY_HINT = "adults_only"
# Members who have all three of these forbidden are almost certainly seeking
# family-safe venues — block adults-only events for them even if they haven't
# explicitly written "adults_only" in their forbidden list (LLM may not always
# extract it, especially for older parsed fixtures).
_NIGHTLIFE_SAFE_SIGNAL = frozenset({"nightclub", "bar", "club"})


def _adults_only_ok(hard: ParsedHardConstraints, attrs: EventAttributes) -> bool:
    """Block adults-only events (18+, kink, fetish) for members who need safe venues."""
    if _ADULTS_ONLY_HINT not in {h.lower() for h in attrs.type_hints}:
        return True
    forbidden_l = {f.lower().strip() for f in hard.forbidden_event_categories}
    if _ADULTS_ONLY_HINT in forbidden_l:
        return False
    # Safety net: member has banned ≥2 nightlife venue types → family-safe context
    if len(forbidden_l & _NIGHTLIFE_SAFE_SIGNAL) >= 2:
        return False
    return True


_SMOKE_CONTRADICTIONS = {"smoking", "smoking_area", "smoking_lounge", "pusacka"}
_PET_CONTRADICTIONS = {"no_pets", "no_dogs", "pets_not_allowed"}

def _venue_requirements_ok(
    hard: ParsedHardConstraints,
    event: NormalizedEvent,
    attrs: EventAttributes,
) -> bool:
    if not hard.venue_requirements:
        return True
    reqs = {r.lower() for r in hard.venue_requirements}
    all_tags = {t.lower() for t in event.tags} | {h.lower() for h in attrs.type_hints}

    if "smoke_free" in reqs and bool(_SMOKE_CONTRADICTIONS & all_tags):
        return False
    if "pet_friendly" in reqs and bool(_PET_CONTRADICTIONS & all_tags):
        return False
    return True


_GLUTEN_CONTRADICTIONS = {"gluten", "breaded", "pasta_only"}
_NUT_CONTRADICTIONS = {"nut_bar", "peanut_heavy"}

def _dietary_restrictions_ok(
    hard: ParsedHardConstraints,
    event: NormalizedEvent,
    attrs: EventAttributes,
) -> bool:
    if not hard.dietary_restrictions:
        return True
    restrictions = {r.lower() for r in hard.dietary_restrictions}
    all_tags = {t.lower() for t in event.tags} | {h.lower() for h in attrs.type_hints}

    if "gluten_free" in restrictions and bool(_GLUTEN_CONTRADICTIONS & all_tags):
        return False
    if "nut_free" in restrictions and bool(_NUT_CONTRADICTIONS & all_tags):
        return False
    return True


def _forbidden_categories_ok(
    hard: ParsedHardConstraints,
    event: NormalizedEvent,
    attrs: EventAttributes,
) -> bool:
    if not hard.forbidden_event_categories:
        return True
    tokens = {raw.lower().strip() for raw in hard.forbidden_event_categories if raw.strip()}
    signals = [t.lower() for t in event.tags] + [h.lower() for h in attrs.type_hints]
    return not any(
        token in signal or signal in token
        for token in tokens
        for signal in signals
    )

from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.rank.group_rank import rank_by_least_misery
from belgrade_recommender.schemas.parsed_preferences import (
    ParsedHardConstraints,
    ParsedSoftPreferences,
    ParsedUserPreferences,
)


def _event(event_id: str, desc: str, tags: list[str]) -> NormalizedEvent:
    return NormalizedEvent(
        event_id=event_id,
        row_index=0,
        link=f"https://example.com/{event_id}",
        source_channel="c",
        tags=tags,
        event_description_raw="",
        event_description_resolved=desc,
        ru_event_description_raw="",
        ru_event_description_resolved="",
        include_in_index=True,
        exclude_reason=None,
    )


def test_rank_by_least_misery_prefers_higher_minimum() -> None:
    """When one event is weak for a picky member, another with higher min should rank first."""

    picky = ParsedUserPreferences(
        summary_one_line="",
        hard_constraints=ParsedHardConstraints(),
        soft_preferences=ParsedSoftPreferences(liked_types=["exhibition"], noise_tolerance="low"),
    )
    easy = ParsedUserPreferences(
        summary_one_line="",
        hard_constraints=ParsedHardConstraints(),
        soft_preferences=ParsedSoftPreferences(liked_types=["outdoor"], noise_tolerance="high"),
    )
    members = [picky, easy]

    ev_weak = _event("a", "Outdoor hike day", tags=["outdoor", "nature"])
    ev_strong = _event("b", "Free museum exhibition opening", tags=["exhibition", "museum"])

    # Slightly higher semantic score on the culture event so the picky member's
    # minimum soft score beats the outdoor-first pairing (orthogonal likes tie on min otherwise).
    candidates = [(ev_weak, 0.85), (ev_strong, 0.88)]
    ranked = rank_by_least_misery(candidates, members)
    assert ranked[0][0].event_id == "b"
    assert ranked[1][0].event_id == "a"

from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.retrieve.hard_filters import passes_all_hard_constraints, passes_hard_constraints
from belgrade_recommender.schemas.event_attributes import EventAttributes
from belgrade_recommender.schemas.parsed_preferences import ParsedHardConstraints


def _event() -> NormalizedEvent:
    return NormalizedEvent(
        event_id="e1",
        row_index=0,
        link="https://t.me/x/1",
        source_channel="c",
        tags=["techno"],
        event_description_raw="",
        event_description_resolved="Techno night",
        ru_event_description_raw="",
        ru_event_description_resolved="",
        include_in_index=True,
        exclude_reason=None,
    )


def test_forbidden_techno_fails() -> None:
    hard = ParsedHardConstraints(forbidden_event_categories=["techno"])
    attrs = EventAttributes(
        event_id="e1",
        type_hints=["techno"],
        price_free_signal=False,
    )
    assert passes_hard_constraints(hard, _event(), attrs) is False


def test_belgrade_user_rejects_novi_sad_hint() -> None:
    hard = ParsedHardConstraints(city="belgrade")
    ev = _event()
    attrs = EventAttributes(event_id="e1", city_hint="novi_sad")
    assert passes_hard_constraints(hard, ev, attrs) is False


def test_must_be_free_rejects_paid_signal() -> None:
    hard = ParsedHardConstraints(must_be_free=True)
    ev = _event()
    attrs = EventAttributes(event_id="e1", price_paid_signal=True, price_amount_rsd=500)
    assert passes_hard_constraints(hard, ev, attrs) is False


def test_must_be_free_accepts_free_signal() -> None:
    hard = ParsedHardConstraints(must_be_free=True)
    ev = _event()
    attrs = EventAttributes(event_id="e1", price_free_signal=True)
    assert passes_hard_constraints(hard, ev, attrs) is True


def test_passes_all_requires_every_member() -> None:
    h_free = ParsedHardConstraints(must_be_free=True)
    h_belgrade = ParsedHardConstraints(city="belgrade")
    ev = _event()
    attrs_ok = EventAttributes(event_id="e1", price_free_signal=True, city_hint="belgrade")
    assert passes_all_hard_constraints([h_free, h_belgrade], ev, attrs_ok) is True

    attrs_paid = EventAttributes(event_id="e1", price_paid_signal=True, city_hint="belgrade")
    assert passes_all_hard_constraints([h_free, h_belgrade], ev, attrs_paid) is False

from __future__ import annotations

import pytest

from belgrade_recommender.extract.attributes import (
    extract_event_attributes,
    extract_event_attributes_maybe_lazy_gemini,
)
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.schemas.event_attributes import EventAttributes


def _event_unknown_noise() -> NormalizedEvent:
    return NormalizedEvent(
        event_id="lazy_test_1",
        row_index=0,
        link="https://example.com/x",
        source_channel="c",
        tags=["misc_unknown_tag"],
        event_description_raw="",
        event_description_resolved="Something vague happening somewhere.",
        ru_event_description_raw="",
        ru_event_description_resolved="",
        include_in_index=True,
        exclude_reason=None,
    )


def test_lazy_disabled_matches_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USE_LAZY_GEMINI_EVENT_ATTRIBUTES", raising=False)
    ev = _event_unknown_noise()
    assert extract_event_attributes_maybe_lazy_gemini(ev).model_dump() == extract_event_attributes(ev).model_dump()


def test_lazy_merges_when_env_and_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_LAZY_GEMINI_EVENT_ATTRIBUTES", "1")

    def stub_gemini(event: NormalizedEvent) -> EventAttributes:
        return EventAttributes(
            event_id=event.event_id,
            price_free_signal=False,
            price_paid_signal=True,
            price_amount_rsd=500,
            price_amount_eur=None,
            city_hint="belgrade",
            date_snippets=["June 1"],
            type_hints=["techno"],
            noise_level_hint="high",
            outdoor_hint=True,
            extraction_method="gemini_v1",
        )

    monkeypatch.setattr(
        "belgrade_recommender.llm.gemini_event_attributes.parse_event_attributes_with_gemini",
        stub_gemini,
    )

    ev = _event_unknown_noise()
    merged = extract_event_attributes_maybe_lazy_gemini(ev)
    assert merged.noise_level_hint == "high"
    assert "techno" in merged.type_hints
    assert merged.extraction_method == "rules_v1_gemini_lazy"
    # price signals stay from rules (stub would have paid — we intentionally keep base prices)
    base = extract_event_attributes(ev)
    assert merged.price_free_signal == base.price_free_signal
    assert merged.price_paid_signal == base.price_paid_signal
    assert merged.price_amount_rsd == base.price_amount_rsd

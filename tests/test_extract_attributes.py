from belgrade_recommender.extract.attributes import extract_event_attributes
from belgrade_recommender.ingest.models import NormalizedEvent


def test_extract_hiking_free_belgrade() -> None:
    event = NormalizedEvent(
        event_id="abc",
        row_index=0,
        link="https://t.me/x/1",
        source_channel="test",
        tags=["hiking", "nature", "freeentry"],
        event_description_raw="",
        event_description_resolved=(
            "Easy hiking near Belgrade on Sunday. Free entry. Meet at 10:00 on 18.04.2026. "
            "Bring water."
        ),
        ru_event_description_raw="",
        ru_event_description_resolved="",
        include_in_index=True,
        exclude_reason=None,
    )
    attrs = extract_event_attributes(event)
    assert attrs.price_free_signal is True
    assert attrs.city_hint == "belgrade"
    assert "hiking" in attrs.type_hints
    assert attrs.outdoor_hint is True

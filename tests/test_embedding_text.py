from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.retrieve.text import build_embedding_text


def test_build_embedding_text_includes_channel_tags_description() -> None:
    event = NormalizedEvent(
        event_id="x",
        row_index=0,
        link="https://example.com/e",
        source_channel="test_ch",
        tags=["jazz", "concert"],
        event_description_raw="",
        event_description_resolved="Evening jazz at the club.",
        ru_event_description_raw="",
        ru_event_description_resolved="",
        include_in_index=True,
        exclude_reason=None,
    )
    text = build_embedding_text(event)
    assert "test_ch" in text
    assert "jazz" in text
    assert "Evening jazz" in text

import json

from belgrade_recommender.ingest.pipeline import parse_jsonl_line


def test_parse_jsonl_line_full_record() -> None:
    raw = {
        "link": "https://t.me/test/1",
        "source_channel": "test_channel",
        "tags": "#Music,#FreeEntry",
        "links": {"PLACEHOLDER_1": {"link_title": "Venue A", "link_address": "https://maps.example/a"}},
        "event_description": (
            "Concert at ${PLACEHOLDER_1}. Free entry. Doors open 19:00, program includes local support act "
            "and headliner until 22:30. Tickets at the venue; accessible by public transport from city centre."
        ),
        "ru_event_description": "",
    }
    line = json.dumps(raw)
    event = parse_jsonl_line(0, line)
    assert event is not None
    assert event.link == "https://t.me/test/1"
    assert event.source_channel == "test_channel"
    assert event.tags == ["music", "freeentry"]
    assert "${PLACEHOLDER_1}" not in event.event_description_resolved
    assert "Venue A" in event.event_description_resolved
    assert event.include_in_index is True
    assert event.exclude_reason is None


def test_parse_jsonl_line_invalid_json_returns_none() -> None:
    assert parse_jsonl_line(0, "not json") is None

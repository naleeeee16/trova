from __future__ import annotations

from pathlib import Path

from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.storage.sqlite_store import bulk_load_normalized_jsonl, load_all_normalized_events


def _one_event() -> NormalizedEvent:
    return NormalizedEvent(
        event_id="e_sql_1",
        row_index=0,
        link="https://example.com/1",
        source_channel="c",
        tags=["exhibition"],
        event_description_raw="",
        event_description_resolved="Free photo exhibition downtown Belgrade",
        ru_event_description_raw="",
        ru_event_description_resolved="",
        include_in_index=True,
        exclude_reason=None,
    )


def test_sqlite_roundtrip(tmp_path: Path) -> None:
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text(_one_event().model_dump_json() + "\n", encoding="utf-8")
    db = tmp_path / "corpus.sqlite"
    n = bulk_load_normalized_jsonl(jsonl, db, replace=True)
    assert n == 1
    loaded = load_all_normalized_events(db)
    assert len(loaded) == 1
    assert loaded[0].event_id == "e_sql_1"
    assert loaded[0].tags == ["exhibition"]

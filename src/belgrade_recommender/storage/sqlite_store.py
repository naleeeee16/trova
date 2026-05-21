"""SQLite: structured rows for events + extracted attributes; full row also in payload_json."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from belgrade_recommender.extract.attributes import extract_event_attributes
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.schemas.event_attributes import EventAttributes


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS normalized_events (
            event_id TEXT PRIMARY KEY,
            row_index INTEGER NOT NULL,
            include_in_index INTEGER NOT NULL,
            exclude_reason TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS event_attributes (
            event_id TEXT PRIMARY KEY,
            city_hint TEXT NOT NULL,
            noise_level_hint TEXT NOT NULL,
            price_free_signal INTEGER NOT NULL,
            price_paid_signal INTEGER NOT NULL,
            price_amount_rsd INTEGER,
            price_amount_eur INTEGER,
            type_hints_json TEXT NOT NULL,
            outdoor_hint INTEGER,
            extraction_method TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES normalized_events(event_id)
        );
        CREATE INDEX IF NOT EXISTS idx_norm_include ON normalized_events(include_in_index);
        """,
    )
    conn.commit()


def bulk_load_normalized_jsonl(
    jsonl_path: Path,
    db_path: Path,
    *,
    replace: bool = True,
) -> int:
    """
    Read ``NormalizedEvent`` lines from JSONL, write ``normalized_events`` + ``event_attributes``.

    If ``replace`` is True, existing tables are cleared first.
    Returns number of rows inserted.
    """

    from belgrade_recommender.retrieve.service import load_normalized_events_jsonl

    events = load_normalized_events_jsonl(jsonl_path)
    conn = _connect(db_path)
    try:
        init_schema(conn)
        if replace:
            conn.execute("DELETE FROM event_attributes")
            conn.execute("DELETE FROM normalized_events")
        cur = conn.cursor()
        n = 0
        for ev in events:
            payload = ev.model_dump_json()
            cur.execute(
                """
                INSERT INTO normalized_events (event_id, row_index, include_in_index, exclude_reason, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ev.event_id,
                    ev.row_index,
                    1 if ev.include_in_index else 0,
                    ev.exclude_reason,
                    payload,
                ),
            )
            attrs = extract_event_attributes(ev)
            _insert_attributes(cur, attrs)
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()


def _insert_attributes(cur: sqlite3.Cursor, attrs: EventAttributes) -> None:
    outdoor_val: int | None
    if attrs.outdoor_hint is None:
        outdoor_val = None
    else:
        outdoor_val = 1 if attrs.outdoor_hint else 0
    cur.execute(
        """
        INSERT INTO event_attributes (
            event_id, city_hint, noise_level_hint,
            price_free_signal, price_paid_signal, price_amount_rsd, price_amount_eur,
            type_hints_json, outdoor_hint, extraction_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            attrs.event_id,
            attrs.city_hint,
            attrs.noise_level_hint,
            1 if attrs.price_free_signal else 0,
            1 if attrs.price_paid_signal else 0,
            attrs.price_amount_rsd,
            attrs.price_amount_eur,
            json.dumps(attrs.type_hints),
            outdoor_val,
            attrs.extraction_method,
        ),
    )


def load_all_normalized_events(db_path: Path) -> list[NormalizedEvent]:
    """Load corpus from SQLite (same order as ``row_index``)."""

    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT payload_json FROM normalized_events ORDER BY row_index ASC",
        )
        out: list[NormalizedEvent] = []
        for (payload,) in cur.fetchall():
            try:
                out.append(NormalizedEvent.model_validate_json(payload))
            except Exception:
                continue
        return out
    finally:
        conn.close()


def load_indexable_event_ids(db_path: Path) -> list[str]:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT event_id FROM normalized_events WHERE include_in_index = 1 ORDER BY row_index ASC",
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

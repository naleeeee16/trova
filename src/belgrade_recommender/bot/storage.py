"""SQLite request log — one row per completed /find session."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("bot.storage")

_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "bot_requests.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    chat_id      INTEGER NOT NULL,
    lang         TEXT    NOT NULL,
    n_users      INTEGER NOT NULL,
    user_inputs  TEXT    NOT NULL,
    user_tags    TEXT    NOT NULL,
    n_results    INTEGER NOT NULL,
    results      TEXT    NOT NULL,
    conflicts    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id          TEXT PRIMARY KEY,
    link              TEXT NOT NULL,
    source_channel    TEXT NOT NULL,
    tags              TEXT NOT NULL,
    description       TEXT NOT NULL,
    description_ru    TEXT NOT NULL,
    include_in_index  INTEGER NOT NULL,
    exclude_reason    TEXT
);
"""


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.executescript(_CREATE_SQL)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
    if "user_tags" not in cols:
        conn.execute("ALTER TABLE requests ADD COLUMN user_tags TEXT NOT NULL DEFAULT '[]'")
        conn.commit()
    return conn


def _build_user_tags(user_inputs: dict[int, str], prefs: list | None) -> list[dict]:
    tags = []
    for i, uid in enumerate(user_inputs):
        p = prefs[i] if prefs and i < len(prefs) else None
        entry: dict = {"user_id": str(uid)}
        if p is not None:
            soft, hard = p.soft_preferences, p.hard_constraints
            entry["liked_types"]              = list(getattr(soft, "liked_types", []) or [])
            entry["noise"]                    = getattr(soft, "noise_tolerance", None)
            entry["ok_paid_eur"]              = getattr(soft, "ok_paid_eur", None)
            entry["preferred_time_windows"]   = list(getattr(soft, "preferred_time_windows", []) or [])
            entry["city"]                     = getattr(hard, "city", None)
            entry["must_be_free"]             = getattr(hard, "must_be_free", None)
            entry["must_be_free_or_budget"]   = getattr(hard, "must_be_free_or_under_budget", None)
            entry["budget_max_rsd"]           = getattr(hard, "budget_max_rsd", None)
            entry["earliest_start_hour"]      = getattr(hard, "earliest_start_hour_weekday", None)
            entry["latest_end_hour"]          = getattr(hard, "latest_end_hour", None)
            entry["preferred_days"]           = list(getattr(hard, "preferred_days_of_week", []) or [])
            entry["forbidden"]                = list(getattr(hard, "forbidden_event_categories", []) or [])
            entry["venue_requirements"]       = list(getattr(hard, "venue_requirements", []) or [])
            entry["dietary_restrictions"]     = list(getattr(hard, "dietary_restrictions", []) or [])
        tags.append(entry)
    return tags


def _build_result_entry(
    rank: int,
    ev,
    sem: float,
    mn: float,
    mean: float,
    reason: str,
    attrs,
    prefs: list | None,
    liked_vecs_list: list | None,
    ev_vec,
) -> dict:
    entry: dict = {
        "rank":        rank,
        "event_id":    ev.event_id,
        "title":       (ev.event_description_resolved or "")[:120].replace("\n", " "),
        "link":        ev.link or "",
        "raw_tags":    list(ev.tags or []),
        "cosine_sem":  round(float(sem),  4),
        "min_soft":    round(float(mn),   4),
        "mean_soft":   round(float(mean), 4),
        "group_score": f"{int(mn * 100)}%",
        "reason":      reason,
    }
    if attrs is not None:
        entry["type_hints"]  = list(getattr(attrs, "type_hints", []) or [])
        entry["noise_level"] = getattr(attrs, "noise_level_hint", None)
        entry["outdoor"]     = getattr(attrs, "outdoor_hint", None)
        entry["city"]        = getattr(attrs, "city_hint", None)
        entry["price_free"]  = getattr(attrs, "price_free_signal", None)
        entry["price_rsd"]   = getattr(attrs, "price_amount_rsd", None)
        entry["price_eur"]   = getattr(attrs, "price_amount_eur", None)

    # Per-user cosine breakdown (mirrors the pipeline log)
    if ev_vec is not None and prefs:
        import numpy as np
        ev_norm = float(np.linalg.norm(ev_vec)) + 1e-9
        user_scores = []
        for ui, p in enumerate(prefs):
            lv = liked_vecs_list[ui] if liked_vecs_list and ui < len(liked_vecs_list) else None
            liked_tokens = list(getattr(p.soft_preferences, "liked_types", []) or [])
            cosines: dict[str, float] = {}
            if lv is not None and len(lv) > 0 and liked_tokens:
                lt_norms = np.linalg.norm(lv, axis=1) + 1e-9
                for tok, c in zip(liked_tokens, (lv @ ev_vec) / (lt_norms * ev_norm)):
                    cosines[tok] = round(float(c), 4)
            user_scores.append({
                "user": ui + 1,
                "noise": getattr(p.soft_preferences, "noise_tolerance", None),
                "liked_cosines": cosines,
            })
        entry["user_scores"] = user_scores

    return entry


def save_request(
    *,
    chat_id: int,
    lang: str,
    user_inputs: dict[int, str],
    ranked: list,
    conflicts: list[str],
    reasons: list[str],
    prefs: list | None = None,
    attrs_map: dict | None = None,
    liked_vecs_list: list | None = None,
    event_vecs: dict | None = None,
) -> None:
    user_tags = _build_user_tags(user_inputs, prefs)

    results_payload = [
        _build_result_entry(
            rank=i + 1,
            ev=ev,
            sem=sem,
            mn=mn,
            mean=mean,
            reason=reasons[i] if i < len(reasons) else "",
            attrs=(attrs_map or {}).get(ev.event_id),
            prefs=prefs,
            liked_vecs_list=liked_vecs_list,
            ev_vec=(event_vecs or {}).get(ev.event_id),
        )
        for i, (ev, sem, mn, mean) in enumerate(ranked)
    ]

    try:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO requests
                (timestamp, chat_id, lang, n_users, user_inputs, user_tags,
                 n_results, results, conflicts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                chat_id,
                lang,
                len(user_inputs),
                json.dumps({str(k): v for k, v in user_inputs.items()}, ensure_ascii=False),
                json.dumps(user_tags, ensure_ascii=False),
                len(ranked),
                json.dumps(results_payload, ensure_ascii=False),
                json.dumps(conflicts, ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()
        log.info("Saved request to DB: chat_id=%d n_results=%d", chat_id, len(ranked))
    except Exception:
        log.warning("Failed to save request to DB", exc_info=True)


def populate_events_table(events: list) -> None:
    """Upsert all normalized events into the events table. Called once at startup."""
    try:
        conn = _connect()
        conn.executemany(
            """
            INSERT INTO events
                (event_id, link, source_channel, tags, description, description_ru,
                 include_in_index, exclude_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                link             = excluded.link,
                source_channel   = excluded.source_channel,
                tags             = excluded.tags,
                description      = excluded.description,
                description_ru   = excluded.description_ru,
                include_in_index = excluded.include_in_index,
                exclude_reason   = excluded.exclude_reason
            """,
            [
                (
                    ev.event_id,
                    ev.link or "",
                    ev.source_channel or "",
                    json.dumps(ev.tags or [], ensure_ascii=False),
                    (ev.event_description_resolved or ev.event_description_raw or ""),
                    (ev.ru_event_description_resolved or ev.ru_event_description_raw or ""),
                    int(ev.include_in_index),
                    ev.exclude_reason,
                )
                for ev in events
            ],
        )
        conn.commit()
        conn.close()
        log.info("Events table populated: %d rows", len(events))
    except Exception:
        log.warning("Failed to populate events table", exc_info=True)

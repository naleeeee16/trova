from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_health() -> None:
    from belgrade_recommender.api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_single_rejects_path_outside_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("BELGRADE_RECOMMENDER_ROOT", str(tmp_path))
    from belgrade_recommender.api.main import app

    client = TestClient(app)
    payload = {
        "events_path": "/etc/passwd",
        "fixture_user_id": "ana",
        "use_hard": False,
        "top_k": 3,
    }
    response = client.post("/v1/recommendations/single", json=payload)
    assert response.status_code == 400


def test_single_unknown_user(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("BELGRADE_RECOMMENDER_ROOT", str(tmp_path))
    (tmp_path / "fixtures").mkdir()
    users = [{"user_id": "x", "preferences_plain_text": "test", "hard_constraints": {}, "soft_preferences": {}}]
    (tmp_path / "fixtures" / "synthetic_users.json").write_text(
        json.dumps({"users": users, "groups": []}),
        encoding="utf-8",
    )
    (tmp_path / "fixtures" / "events_smoke.jsonl").write_text("", encoding="utf-8")

    from belgrade_recommender.api.main import app

    client = TestClient(app)
    payload = {
        "events_path": "fixtures/events_smoke.jsonl",
        "fixture_user_id": "ana",
        "use_hard": False,
        "top_k": 3,
    }
    response = client.post("/v1/recommendations/single", json=payload)
    assert response.status_code == 400

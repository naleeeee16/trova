import json
from pathlib import Path


def test_synthetic_users_fixture_schema() -> None:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_users.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("version") == 1
    users = data["users"]
    assert len(users) == 4
    ids = {u["user_id"] for u in users}
    assert ids == {"ana", "marko", "jelena", "stefan"}
    for user in users:
        assert "preferences_plain_text" in user
        assert "hard_constraints" in user
        assert "soft_preferences" in user
    groups = data["groups"]
    assert len(groups) >= 1
    assert "member_user_ids" in groups[0]

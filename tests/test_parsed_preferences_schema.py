"""Golden JSON tests for ParsedUserPreferences (no live Gemini)."""

from belgrade_recommender.schemas.parsed_preferences import ParsedUserPreferences


def test_model_validate_json_roundtrip() -> None:
    payload = {
        "summary_one_line": "Quiet cultural events in Belgrade with a modest budget.",
        "hard_constraints": {
            "city": "belgrade",
            "budget_max_rsd": 2000,
            "must_be_free": None,
            "must_be_free_or_under_budget": True,
            "earliest_start_hour_weekday": None,
            "latest_end_hour": 23,
            "forbidden_event_categories": ["techno", "rave"],
        },
        "soft_preferences": {
            "liked_types": ["exhibition", "lecture"],
            "noise_tolerance": "low",
            "ok_paid_eur": None,
            "preferred_time_windows": ["weekend_afternoon"],
        },
    }
    parsed = ParsedUserPreferences.model_validate(payload)
    again = ParsedUserPreferences.model_validate_json(parsed.model_dump_json())
    assert again.hard_constraints.budget_max_rsd == 2000
    assert again.soft_preferences.noise_tolerance == "low"

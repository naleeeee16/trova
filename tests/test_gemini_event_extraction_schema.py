from belgrade_recommender.schemas.gemini_event_extraction import GeminiEventAttributeExtraction


def test_gemini_extraction_schema_roundtrip() -> None:
    payload = {
        "price_free_signal": True,
        "price_paid_signal": False,
        "price_amount_rsd": None,
        "price_amount_eur": None,
        "city_hint": "belgrade",
        "date_snippets": ["May 9"],
        "type_hints": ["exhibition", "festival"],
        "noise_level_hint": "low",
        "outdoor_hint": None,
    }
    g = GeminiEventAttributeExtraction.model_validate(payload)
    assert g.city_hint == "belgrade"
    assert "exhibition" in g.type_hints


def test_json_roundtrip() -> None:
    raw = '{"price_free_signal": false, "price_paid_signal": true, "city_hint": "unknown", "type_hints": ["techno"]}'
    g = GeminiEventAttributeExtraction.model_validate_json(raw)
    assert g.type_hints == ["techno"]

from belgrade_recommender.ingest.placeholders import resolve_placeholders


def test_resolve_placeholders_replaces_with_title() -> None:
    text = "Meet at ${PLACEHOLDER_1} on Sunday."
    links = {"PLACEHOLDER_1": {"link_title": "Dorćol Platz", "link_address": "https://example.com"}}
    assert resolve_placeholders(text, links) == "Meet at Dorćol Platz on Sunday."


def test_resolve_placeholders_missing_key_becomes_empty() -> None:
    text = "Location ${PLACEHOLDER_99}"
    assert resolve_placeholders(text, {}) == "Location "


def test_resolve_placeholders_empty_text() -> None:
    assert resolve_placeholders("", {"PLACEHOLDER_1": {"link_title": "x"}}) == ""

from belgrade_recommender.extract import rules


def test_detect_free_entry() -> None:
    assert rules.detect_free_entry("Something something Free entry before 8pm.") is True
    assert rules.detect_free_entry("Tickets from 2000 RSD") is False


def test_max_rsd_amount() -> None:
    assert rules.max_rsd_amount("Price 2000 RSD per person") == 2000
    assert rules.max_rsd_amount("from 2000 RSD and 3000 RSD") == 3000


def test_max_eur_amount() -> None:
    assert rules.max_eur_amount("Tour costs 80 EUR") == 80


def test_detect_city_hint() -> None:
    assert rules.detect_city_hint("Event in Belgrade centre") == "belgrade"
    assert rules.detect_city_hint("Starts in Novi Sad") == "novi_sad"


def test_extract_date_snippets() -> None:
    snippets = rules.extract_date_snippets("Join us on 05.04.2026 at the venue.")
    assert any("05.04.2026" in s for s in snippets)

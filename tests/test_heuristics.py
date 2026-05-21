from belgrade_recommender.ingest.heuristics import classify_for_retrieval_index


def test_too_short_excluded() -> None:
    inc, reason = classify_for_retrieval_index(
        event_description_resolved="x" * 40,
        tags=["music"],
        event_description_raw="",
    )
    assert inc is False
    assert reason == "too_short"


def test_good_morning_excluded() -> None:
    text = "y" * 100
    inc, reason = classify_for_retrieval_index(
        event_description_resolved=f"Good morning! {text}",
        tags=[],
        event_description_raw="",
    )
    assert inc is False
    assert reason is not None
    assert reason.startswith("keyword:")


def test_promo_tag_excluded() -> None:
    body = "y" * 100
    inc, reason = classify_for_retrieval_index(
        event_description_resolved=f"Some event description here. {body}",
        tags=["concert", "promo"],
        event_description_raw="",
    )
    assert inc is False
    assert reason == "tag:promo"


def test_normal_event_included() -> None:
    text = (
        "Join us for an outdoor jazz evening in Belgrade city centre. "
        "Free entry, starts 18:00, ends around 21:00. Family friendly."
    )
    inc, reason = classify_for_retrieval_index(
        event_description_resolved=text,
        tags=["jazz", "outdoor"],
        event_description_raw=text,
    )
    assert inc is True
    assert reason is None

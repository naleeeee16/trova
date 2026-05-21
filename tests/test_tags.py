from belgrade_recommender.ingest.tags import normalize_tags


def test_normalize_tags_splits_and_lowercases() -> None:
    assert normalize_tags("#Hiking,#Nature,#Hiking") == ["hiking", "nature"]


def test_normalize_tags_empty() -> None:
    assert normalize_tags("") == []
    assert normalize_tags("  , , ") == []

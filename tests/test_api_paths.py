from __future__ import annotations

import pytest

from belgrade_recommender.api.paths import resolve_under_repo


def test_resolve_rejects_escape(tmp_path) -> None:
    root = tmp_path.resolve()
    (root / "safe.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_under_repo(root, "../../../../etc/passwd", must_be_file=True, must_be_dir=False)


def test_resolve_relative_file(tmp_path) -> None:
    root = tmp_path.resolve()
    (root / "data").mkdir()
    f = root / "data" / "a.jsonl"
    f.write_text("{}", encoding="utf-8")
    got = resolve_under_repo(root, "data/a.jsonl", must_be_file=True, must_be_dir=False)
    assert got == f

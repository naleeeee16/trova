"""Resolve API path strings to files under a trusted repo root (path traversal guard)."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Root directory for resolving ``events_path`` / fixtures (default: process cwd)."""

    return Path(os.environ.get("BELGRADE_RECOMMENDER_ROOT", ".")).resolve()


def resolve_under_repo(root: Path, raw: str, *, must_be_file: bool, must_be_dir: bool) -> Path:
    raw = raw.strip()
    if not raw:
        raise ValueError("path is empty")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        if not resolved.is_relative_to(root):
            raise ValueError("path must stay under repo root")
    except ValueError as exc:
        raise ValueError("invalid or unsafe path") from exc
    if must_be_file and not resolved.is_file():
        raise ValueError("file not found")
    if must_be_dir and not resolved.is_dir():
        raise ValueError("directory not found")
    return resolved

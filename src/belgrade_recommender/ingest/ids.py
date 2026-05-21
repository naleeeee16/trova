from __future__ import annotations

import hashlib


def stable_event_id(link: str) -> str:
    """Stable 16-hex id from link (primary natural key in the dataset)."""

    normalized = link.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]

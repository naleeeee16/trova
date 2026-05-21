from __future__ import annotations


def normalize_tags(tags: str) -> list[str]:
    """Split comma-separated hashtags, strip #, lowercase, dedupe while preserving order."""

    if not tags or not str(tags).strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in str(tags).split(","):
        token = part.strip()
        if not token:
            continue
        if token.startswith("#"):
            token = token[1:].strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out

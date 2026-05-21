from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER_PATTERN = re.compile(r"\$\{PLACEHOLDER_(\d+)\}")


def resolve_placeholders(text: str, links: dict[str, Any] | None) -> str:
    """If the descriptioon is empty"""

    if not text:
        return ""
    if not links:
        return _PLACEHOLDER_PATTERN.sub("", text)

    def repl(match: re.Match[str]) -> str:
        key = f"PLACEHOLDER_{match.group(1)}"
        block = links.get(key)
        if not isinstance(block, dict):
            return ""
        title = block.get("link_title")
        if title is None:
            return ""
        return str(title).strip()

    return _PLACEHOLDER_PATTERN.sub(repl, text)

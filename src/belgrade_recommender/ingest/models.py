from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RawEventRecord(BaseModel):
    """Single row as stored in events_dataset.jsonl."""

    link: str
    source_channel: str
    tags: str = ""
    links: dict[str, Any] = Field(default_factory=dict)
    ru_event_description: str = ""
    event_description: str = ""

    @field_validator("link", "source_channel", mode="before")
    @classmethod
    def strip_str(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()


class NormalizedEvent(BaseModel):
    """Cleaned record ready for indexing and feature extraction."""

    event_id: str
    row_index: int
    link: str
    source_channel: str
    tags: list[str] = Field(default_factory=list)
    event_description_raw: str = ""
    event_description_resolved: str = ""
    ru_event_description_raw: str = ""
    ru_event_description_resolved: str = ""
    include_in_index: bool = True
    exclude_reason: str | None = None

"""Structured event fields derived from text/tags (Phase 2).

Field names and semantics are chosen to align with preference / constraint
concepts (budget, city, type, noise) for later filtering and scoring.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CityHint = Literal["belgrade", "novi_sad", "serbia_other", "unknown"]
NoiseHint = Literal["low", "medium", "medium_high", "high", "unknown"]


class EventAttributes(BaseModel):
    """Rule-based structured signals for one event (v1 heuristics)."""

    event_id: str
    price_free_signal: bool = False
    price_paid_signal: bool = False
    price_amount_rsd: int | None = Field(default=None, ge=0)
    price_amount_eur: int | None = Field(default=None, ge=0)
    city_hint: CityHint = "unknown"
    date_snippets: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Raw date matches from text (capped at 5).",
    )
    day_of_week_hints: list[str] = Field(
        default_factory=list,
        description="Canonical English day names inferred from event text (monday…sunday).",
    )

    type_hints: list[str] = Field(
        default_factory=list,
        description="Merged snake_case hints from tags + keywords (deduped).",
    )
    noise_level_hint: NoiseHint = "unknown"
    outdoor_hint: bool | None = Field(
        default=None,
        description="True if outdoor-ish tags/text; False if clearly indoor; None if unknown.",
    )
    extraction_method: str = Field(default="rules_v1", description="Audit trail for how fields were produced.")

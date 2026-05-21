"""Subset of event signals produced by Gemini (structured JSON); ``event_id`` is attached after parse."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CityHint = Literal["belgrade", "novi_sad", "serbia_other", "unknown"]
NoiseHint = Literal["low", "medium", "high", "unknown"]


class GeminiEventAttributeExtraction(BaseModel):
    """Same field semantics as ``EventAttributes`` except ``event_id`` / ``extraction_method``."""

    price_free_signal: bool = False
    price_paid_signal: bool = False
    price_amount_rsd: int | None = Field(default=None, ge=0)
    price_amount_eur: int | None = Field(default=None, ge=0)
    city_hint: CityHint = "unknown"
    date_snippets: list[str] = Field(default_factory=list, max_length=5)
    type_hints: list[str] = Field(default_factory=list)
    noise_level_hint: NoiseHint = "unknown"
    outdoor_hint: bool | None = Field(
        default=None,
        description="True outdoor-ish; False clearly indoor; null unknown.",
    )

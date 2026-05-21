"""Structured preference profile produced by LLM (Gemini structured output).

Aligned with the same concepts as fixtures/synthetic_users.json so they can
later match structured event attributes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


NoiseTolerance = Literal["low", "medium", "medium_high", "high", "unknown"]

# Quality/mood adjectives the LLM sometimes produces that don't correspond to
# any event-type token in our vocabulary.  Strip them silently after parsing.
_LIKED_TYPES_BANNED: frozenset[str] = frozenset({
    "romantic", "cozy", "fancy", "intimate", "hidden", "authentic",
    "local", "special", "unique", "relaxed", "nice", "fun", "glamorous",
    "free", "cheap", "safe", "cool", "amazing", "great", "interesting",
    "modern", "trendy", "aesthetic", "underground", "vibe",
})

# Tokens the LLM produces that are not event types at all — drop them before
# embedding so they don't dilute the liked_types signal.
# Vocabulary bridging (turbofolk ↔ folk_traditional, gallery ↔ exhibition, etc.)
# is handled by the embedding model at scoring time and no longer needs to be
# listed here.
_LIKED_TYPES_NORMALIZE: dict[str, str | None] = {
    # seasons / generic time words — not event categories
    "summer":           None,
    "winter":           None,
    "spring":           None,
    "autumn":           None,
    # descriptors with no matching event type in the corpus
    "analog":           None,
    "detox":            None,
    "silent_retreat":   None,
    "neon":             None,
    "hidden_gem":       None,
    "vibes":            None,
    "atmosphere":       None,
    "ambiance":         None,
}


class ParsedHardConstraints(BaseModel):
    """Hard filters: violations should remove or heavily penalize candidates."""

    city: str | None = Field(
        default=None,
        description="e.g. belgrade, belgrade_or_daytrip_serbia, unknown",
    )
    budget_max_rsd: int | None = Field(default=None, ge=0)
    must_be_free: bool | None = None
    must_be_free_or_under_budget: bool | None = None
    earliest_start_hour_weekday: int | None = Field(default=None, ge=0, le=23)
    latest_end_hour: int | None = Field(default=None, ge=0, le=23)
    preferred_days_of_week: list[str] = Field(
        default_factory=list,
        description="Canonical English day names the user prefers (monday…sunday). Empty = no constraint.",
    )
    forbidden_event_categories: list[str] = Field(default_factory=list)
    venue_requirements: list[str] = Field(
        default_factory=list,
        description="Strict venue attribute requirements: smoke_free, non_alcoholic, pet_friendly.",
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="Strict dietary/allergy requirements: vegan, gluten_free, nut_free, raw_vegan, halal.",
    )


class ParsedSoftPreferences(BaseModel):
    """Soft signals: used for scoring / ranking, not hard removal."""

    liked_types: list[str] = Field(default_factory=list)
    noise_tolerance: NoiseTolerance = "unknown"
    ok_paid_eur: bool | None = None
    preferred_time_windows: list[str] = Field(default_factory=list)

    @field_validator("noise_tolerance", mode="before")
    @classmethod
    def _coerce_noise(cls, v: object) -> str:
        return v if v is not None else "unknown"

    @field_validator("liked_types", mode="before")
    @classmethod
    def _normalize_liked_types(cls, v: object) -> list[str]:
        """Strip banned quality adjectives and normalize non-standard tokens.

        Runs after the LLM produces liked_types so downstream scoring always
        operates on the canonical event-vocabulary token set.
        """
        if not isinstance(v, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for raw in v:
            if not isinstance(raw, str):
                continue
            token = raw.strip().lower().replace(" ", "_")
            if not token:
                continue
            # Drop quality adjectives that don't represent event types
            if token in _LIKED_TYPES_BANNED:
                continue
            # Remap to canonical vocabulary; None means "drop this token"
            if token in _LIKED_TYPES_NORMALIZE:
                mapped = _LIKED_TYPES_NORMALIZE[token]
                if mapped is None:
                    continue
                token = mapped
            if token not in seen:
                seen.add(token)
                result.append(token)
        return result


class ParsedUserPreferences(BaseModel):
    """Full structured profile from one user's natural-language preferences."""

    summary_one_line: str = Field(
        default="",
        description="Short neutral summary for UI or logs (no PII beyond what user typed).",
    )
    hard_constraints: ParsedHardConstraints = Field(default_factory=ParsedHardConstraints)
    soft_preferences: ParsedSoftPreferences = Field(default_factory=ParsedSoftPreferences)

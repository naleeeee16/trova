"""Parse natural-language preferences with OpenAI structured JSON output."""

from __future__ import annotations

import logging
import os

from belgrade_recommender.llm.prompts import PREFERENCE_SYSTEM_INSTRUCTION
from belgrade_recommender.schemas.parsed_preferences import ParsedUserPreferences

log = logging.getLogger(__name__)


def parse_preferences_openai(
    preferences_plain_text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> ParsedUserPreferences:
    """
    Call OpenAI with structured output to fill ParsedUserPreferences.

    Environment:
    - OPENAI_API_KEY: required for live calls.
    - OPENAI_MODEL: optional override (default: gpt-4o-mini).
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY for OpenAI preference parsing.")

    text = (preferences_plain_text or "").strip()
    if not text:
        return ParsedUserPreferences()

    resolved_model = (model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = OpenAI(api_key=key)

    log.info("LLM input [preference_parse] model=%s text=%r", resolved_model, text[:120])

    response = client.beta.chat.completions.parse(
        model=resolved_model,
        messages=[
            {"role": "system", "content": PREFERENCE_SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": f"User preference text:\n{text}\n\nExtract structured fields as JSON.",
            },
        ],
        response_format=ParsedUserPreferences,
        temperature=0,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned an empty response for preference parsing.")

    log.info("LLM output [preference_parse] liked_types=%s hard=%s",
             getattr(parsed.soft_preferences, "liked_types", None),
             parsed.hard_constraints.model_dump(exclude_none=True) if parsed.hard_constraints else {})

    return parsed

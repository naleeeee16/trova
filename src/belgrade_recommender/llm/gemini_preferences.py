"""Parse natural-language preferences with Gemini structured JSON output."""

from __future__ import annotations

import importlib.util
import os

from belgrade_recommender.llm.prompts import PREFERENCE_SYSTEM_INSTRUCTION
from belgrade_recommender.schemas.parsed_preferences import ParsedUserPreferences


def _gemini_available() -> bool:
    return importlib.util.find_spec("google.genai") is not None


def parse_preferences_plain_text(
    preferences_plain_text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> ParsedUserPreferences:
    """
    Call Gemini with structured output to fill ParsedUserPreferences.

    Environment:
    - GEMINI_API_KEY: required for live calls.
    """

    if not _gemini_available():
        raise ImportError(
            'Missing optional dependency google-genai. Install with: pip install google-genai',
        )

    from google import genai
    from google.genai import types

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY for Gemini preference parsing.")

    text = (preferences_plain_text or "").strip()
    if not text:
        return ParsedUserPreferences()

    resolved_model = (model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()
    client = genai.Client(api_key=key)

    user_block = f"User preference text:\n{text}\n\nExtract structured fields as JSON."

    config = types.GenerateContentConfig(
        system_instruction=PREFERENCE_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=ParsedUserPreferences,
    )

    response = client.models.generate_content(
        model=resolved_model,
        contents=user_block,
        config=config,
    )

    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned an empty response for preference parsing.")

    return ParsedUserPreferences.model_validate_json(raw)

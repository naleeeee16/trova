"""Structured event attribute extraction via Gemini (complex / messy descriptions)."""

from __future__ import annotations

import importlib.util
import inspect
import os
from typing import Any

from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.llm.prompts import EVENT_ATTRIBUTES_LLM_INSTRUCTION
from belgrade_recommender.schemas.event_attributes import EventAttributes
from belgrade_recommender.schemas.gemini_event_extraction import GeminiEventAttributeExtraction


def _gemini_available() -> bool:
    return importlib.util.find_spec("google.genai") is not None


def _generation_config(types: Any, schema: dict[str, Any]) -> tuple[Any, bool]:
    sig = inspect.signature(types.GenerateContentConfig)
    params: dict[str, Any] = {"response_mime_type": "application/json"}
    used_system_instruction = False
    if "system_instruction" in sig.parameters:
        params["system_instruction"] = EVENT_ATTRIBUTES_LLM_INSTRUCTION
        used_system_instruction = True
    if "response_json_schema" in sig.parameters:
        params["response_json_schema"] = schema
    elif "response_schema" in sig.parameters:
        params["response_schema"] = schema
    else:
        raise RuntimeError(
            "google-genai GenerateContentConfig has no response_json_schema or response_schema; "
            "upgrade: pip install -U 'google-genai>=1.0'",
        )
    return types.GenerateContentConfig(**params), not used_system_instruction


def _event_prompt_block(event: NormalizedEvent) -> str:
    tags = ", ".join(event.tags) if event.tags else "(none)"
    text = (event.event_description_resolved or event.event_description_raw or "").strip()
    return (
        f"event_id (context only, do not echo as root key): {event.event_id}\n"
        f"link: {event.link}\n"
        f"tags: {tags}\n\n"
        f"description:\n{text}\n"
    )


def parse_event_attributes_with_gemini(
    event: NormalizedEvent,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> EventAttributes:
    """
    Fill ``EventAttributes`` using Gemini structured JSON (for messy text where rules underperform).

    Requires ``GEMINI_API_KEY`` and ``pip install -e ".[gemini]"``.
    """

    if not _gemini_available():
        raise ImportError(
            'Missing optional dependency google-genai. Install with: pip install -e ".[gemini]"',
        )

    from google import genai
    from google.genai import types

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini event extraction.",
        )

    resolved_model = (model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()
    client = genai.Client(api_key=key)

    schema = GeminiEventAttributeExtraction.model_json_schema()
    config, need_system_prefix = _generation_config(types, schema)
    user_block = _event_prompt_block(event) + "\nExtract structured fields as JSON (no event_id field)."
    contents = (
        f"{EVENT_ATTRIBUTES_LLM_INSTRUCTION}\n\n{user_block}" if need_system_prefix else user_block
    )

    response = client.models.generate_content(
        model=resolved_model,
        contents=contents,
        config=config,
    )
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned an empty response for event attributes.")

    body = GeminiEventAttributeExtraction.model_validate_json(raw)
    return EventAttributes(
        event_id=event.event_id,
        extraction_method="gemini_v1",
        **body.model_dump(),
    )

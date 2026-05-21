"""Unit tests for Gemini config helper (no google-genai import required)."""

from __future__ import annotations

from typing import Any

from belgrade_recommender.llm import gemini_preferences as gp


def test_generation_config_prefers_response_json_schema() -> None:
    class ConfigCls:
        def __init__(
            self,
            *,
            response_mime_type: str,
            system_instruction: str | None = None,
            response_json_schema: dict[str, Any] | None = None,
        ) -> None:
            self.kw = {
                "response_mime_type": response_mime_type,
                "system_instruction": system_instruction,
                "response_json_schema": response_json_schema,
            }

    class Types:
        GenerateContentConfig = ConfigCls

    schema = {"type": "object", "properties": {}}
    cfg, need_prefix = gp._generation_config(Types, schema)
    assert need_prefix is False
    assert cfg.kw["response_mime_type"] == "application/json"
    assert cfg.kw["response_json_schema"] == schema
    assert cfg.kw["system_instruction"]


def test_generation_config_falls_back_to_response_schema() -> None:
    class ConfigCls:
        def __init__(
            self,
            *,
            response_mime_type: str,
            response_schema: dict[str, Any],
        ) -> None:
            self.kw = {"response_mime_type": response_mime_type, "response_schema": response_schema}

    class Types:
        GenerateContentConfig = ConfigCls

    schema = {"type": "object"}
    cfg, need_prefix = gp._generation_config(Types, schema)
    assert need_prefix is True
    assert cfg.kw["response_schema"] == schema

"""Plain-language explanations for ranked events (post-retrieval, optional Gemini)."""

from __future__ import annotations

import importlib.util
import os

from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.llm.prompts import RANKING_EXPLAIN_INSTRUCTION


def _gemini_available() -> bool:
    return importlib.util.find_spec("google.genai") is not None


def explain_ranking_plain(
    *,
    preferences_context: str,
    ranked_lines: list[str],
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Turn structured preference context + ranked bullet lines into a short user-facing explanation.

    ``preferences_context`` is typically JSON from ``ParsedUserPreferences`` or a short
    combined summary for a group. ``ranked_lines`` are human-readable bullets (title, link, scores).

    Requires ``GEMINI_API_KEY``  ``pip install -e ".[gemini]"``.
    """

    if not _gemini_available():
        raise ImportError(
            'Missing optional dependency google-genai. Install with: pip install -e ".[gemini]"',
        )

    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini explanations.",
        )

    resolved_model = (model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()
    client = genai.Client(api_key=key)

    bullets = "\n".join(ranked_lines) if ranked_lines else "(no ranked events)"
    user_block = (
        "Preference / constraint context (JSON or text):\n"
        f"{preferences_context.strip()}\n\n"
        "Ranked recommendations (best first):\n"
        f"{bullets}\n\n"
        "Write the explanation now."
    )
    contents = f"{RANKING_EXPLAIN_INSTRUCTION}\n\n{user_block}"

    response = client.models.generate_content(
        model=resolved_model,
        contents=contents,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty explanation.")
    return text


def ranked_lines_from_single(
    ranked: list[tuple[NormalizedEvent, float]],
    *,
    max_lines: int = 12,
) -> list[str]:
    """Format (NormalizedEvent, score) rows as stable bullet strings for the explainer."""

    lines: list[str] = []
    for i, (ev, score) in enumerate(ranked[:max_lines], start=1):
        title = (ev.event_description_resolved or "")[:160].replace("\n", " ")
        lines.append(f"{i}. score={score:.4f} | {title} | {ev.link}")
    return lines


def ranked_lines_from_group(
    ranked: list[tuple[NormalizedEvent, float, float, float]],
    *,
    max_lines: int = 12,
) -> list[str]:
    """Format least-misery rows as bullet strings."""

    lines: list[str] = []
    for i, (ev, sem, mn, mean) in enumerate(ranked[:max_lines], start=1):
        title = (ev.event_description_resolved or "")[:160].replace("\n", " ")
        lines.append(
            f"{i}. sem={sem:.4f} min_soft={mn:.4f} mean_soft={mean:.4f} | {title} | {ev.link}",
        )
    return lines

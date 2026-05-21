"""Per-event explanations for ranked results via a single OpenAI call."""

from __future__ import annotations

import json
import logging
import os

from belgrade_recommender.ingest.models import NormalizedEvent

log = logging.getLogger(__name__)


_LANG_NAMES = {
    "srb": "Serbian",
    "rus": "Russian",
    "eng": "English",
}

_LANG_EXAMPLES = {
    "srb": "tiha atmosfera, craft pivo, terasa, besplatno, jaz muzika",
    "rus": "тихая атмосфера, крафтовое пиво, терраса, бесплатно, джаз",
    "eng": "quiet atmosphere, craft beer, terrace, free entry, jazz music",
}

_SCORE_LABEL = {
    "srb": "skor",
    "rus": "оценка",
    "eng": "score",
}

_USER_PROMPT_SUFFIX = {
    "srb": "Napiši kratko objašnjenje za svaki event zašto odgovara grupi.",
    "rus": "Напиши краткое объяснение для каждого события, почему оно подходит группе.",
    "eng": "Write a short explanation for each event why it fits this group.",
}

_GROUP_PREFS_LABEL = {
    "srb": "Preferencije grupe",
    "rus": "Предпочтения группы",
    "eng": "Group preferences",
}

_EVENTS_LABEL = {
    "srb": "Preporučeni eventi",
    "rus": "Рекомендуемые события",
    "eng": "Recommended events",
}


def _build_system_prompt(lang: str) -> str:
    lang_name = _LANG_NAMES.get(lang, "Serbian")
    examples = _LANG_EXAMPLES.get(lang, _LANG_EXAMPLES["srb"])
    return (
        f"You are an assistant for a Belgrade group event recommender.\n"
        f"Given group preferences and a list of recommended events, write ONE short sentence per event\n"
        f"(max 15 words each) in {lang_name} explaining why that event fits this specific group.\n"
        f"Rules:\n"
        f"- Be concrete: mention the matching aspect ({examples}).\n"
        f"- Do NOT repeat the event title — explain the match.\n"
        f"- Match the tone to the group: casual for friends, professional for colleagues, etc.\n"
        f"Return ONLY valid JSON: {{\"reasons\": [\"sentence 1\", \"sentence 2\", ...]}}\n"
        f"The list must have exactly as many items as there are events in the input."
    )


def explain_per_event(
    *,
    preferences_context: str,
    ranked: list[tuple[NormalizedEvent, float, float, float]],
    lang: str = "srb",
    model: str | None = None,
    api_key: str | None = None,
) -> list[str]:
    """
    Returns a list of short reason strings (in the requested language), one per ranked event.
    Falls back to empty strings on any error.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key or not ranked:
        return [""] * len(ranked)

    resolved_model = (model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = OpenAI(api_key=key)

    score_lbl = _SCORE_LABEL.get(lang, "score")
    event_lines = []
    for i, (ev, sem, mn, mean) in enumerate(ranked, 1):
        desc = (ev.event_description_resolved or "")[:150].replace("\n", " ")
        event_lines.append(f"{i}. ({score_lbl} {mn*100:.0f}%) {desc}")

    prefs_label = _GROUP_PREFS_LABEL.get(lang, _GROUP_PREFS_LABEL["eng"])
    events_label = _EVENTS_LABEL.get(lang, _EVENTS_LABEL["eng"])
    suffix = _USER_PROMPT_SUFFIX.get(lang, _USER_PROMPT_SUFFIX["eng"])

    user_block = (
        f"{prefs_label}:\n{preferences_context.strip()}\n\n"
        f"{events_label}:\n" + "\n".join(event_lines) +
        f"\n\n{suffix}"
    )

    log.info(
        "LLM input [explain_per_event] model=%s lang=%s n_events=%d prefs=%r",
        resolved_model, lang, len(ranked), preferences_context[:80],
    )

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": _build_system_prompt(lang)},
                {"role": "user", "content": user_block},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=600,
        )
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)
        reasons = data.get("reasons", [])
    except Exception:
        log.warning("LLM output [explain_per_event] failed — returning empty reasons", exc_info=True)
        return [""] * len(ranked)

    log.info("LLM output [explain_per_event] n_reasons=%d first=%r", len(reasons), reasons[0] if reasons else "")

    # Pad or trim to match ranked length
    reasons = list(reasons) + [""] * len(ranked)
    return reasons[: len(ranked)]


# ---------------------------------------------------------------------------
# Missing-tag explanation — called when no good results were found
# ---------------------------------------------------------------------------

_MISSING_SYSTEM: dict[str, str] = {
    "srb": (
        "Ti si asistent za preporuku događaja u Beogradu. Grupna pretraga nije pronašla "
        "dobre rezultate. Na osnovu preferencija grupe i toga koje specifične teme/interesi "
        "nisu pronađeni, napiši toplo i korisno objašnjenje (2-3 rečenice) na srpskom jeziku. "
        "Objasni šta nije pronađeno i predloži da probaju sa širim terminima ili da ukucaju /find. "
        "Budi konkretan — pomeni specifične interese koji nisu pronađeni."
    ),
    "rus": (
        "Ты ассистент для рекомендации событий в Белграде. Групповой поиск не нашёл хороших совпадений. "
        "На основе предпочтений группы и того, какие конкретные интересы не были найдены, "
        "напиши тёплое и полезное объяснение (2-3 предложения) на русском языке. "
        "Объясни, что не нашлось, и предложи попробовать более широкие термины или снова ввести /find. "
        "Будь конкретным — упомяни конкретные интересы, которые не были найдены."
    ),
    "eng": (
        "You are an assistant for a Belgrade group event recommender. The group search found no good matches. "
        "Given the group's preferences and which specific interests/tags had no matches, "
        "write a warm, helpful 2-3 sentence explanation in English. "
        "Explain what couldn't be found and suggest they try broader terms or use /find again. "
        "Be specific — mention the specific interests that weren't found."
    ),
}

_MISSING_PREFS_LABEL: dict[str, str] = {
    "srb": "Preferencije grupe",
    "rus": "Предпочтения группы",
    "eng": "Group preferences",
}

_MISSING_UNFULFILLED_LABEL: dict[str, str] = {
    "srb": "Interesi bez odgovarajućih događaja",
    "rus": "Интересы без подходящих событий",
    "eng": "Interests with no matching events",
}

_MISSING_CONFLICTS_LABEL: dict[str, str] = {
    "srb": "Suprotstavljene preference",
    "rus": "Противоречивые предпочтения",
    "eng": "Conflicting preferences",
}


def explain_missing_tags(
    *,
    user_summaries: list[str],
    unfulfilled: dict[str, list[str]],
    conflicts: list[str],
    lang: str = "srb",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Return a natural-language explanation of which preference tags had no matching
    events in Belgrade. Falls back to empty string on any error.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return ""

    resolved_model = (model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = OpenAI(api_key=key)

    prefs_label       = _MISSING_PREFS_LABEL.get(lang, _MISSING_PREFS_LABEL["eng"])
    unfulfilled_label = _MISSING_UNFULFILLED_LABEL.get(lang, _MISSING_UNFULFILLED_LABEL["eng"])
    conflicts_label   = _MISSING_CONFLICTS_LABEL.get(lang, _MISSING_CONFLICTS_LABEL["eng"])

    parts = [f"{prefs_label}:"]
    for s in user_summaries:
        parts.append(f"- {s}")

    if unfulfilled:
        parts.append(f"\n{unfulfilled_label}:")
        for user, tags in unfulfilled.items():
            parts.append(f"- {user}: {', '.join(tags)}")

    if conflicts:
        parts.append(f"\n{conflicts_label}:")
        for c in conflicts:
            parts.append(f"- {c}")

    system = _MISSING_SYSTEM.get(lang, _MISSING_SYSTEM["eng"])
    user_text = "\n".join(parts)

    log.info(
        "LLM input [explain_missing_tags] model=%s lang=%s n_unfulfilled=%d n_conflicts=%d",
        resolved_model, lang, len(unfulfilled), len(conflicts),
    )

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            temperature=0.4,
            max_tokens=200,
        )
        result = (response.choices[0].message.content or "").strip()
    except Exception:
        log.warning("LLM output [explain_missing_tags] failed — returning empty", exc_info=True)
        return ""

    log.info("LLM output [explain_missing_tags] length=%d", len(result))
    return result

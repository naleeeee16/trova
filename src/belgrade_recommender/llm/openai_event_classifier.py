"""Batch LLM classification: is a post an attendable public event or not?

Called once during dataset ingestion (not at query time) so the cost is paid
upfront and stored in the normalized JSONL. At query time no AI call is needed.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You classify social media posts from Belgrade event channels.

For each post decide: is it an ATTENDABLE PUBLIC EVENT — something a person
could physically go to (concert, festival, exhibition, restaurant opening
people can visit, market, workshop, party, tour, sports event, etc.)?

Mark is_event=false for:
- Catering service advertisements ("we provide catering", "hire us for your event")
- Blogger / influencer video or review posts ("watch my new video", "check out my review", "link in bio")
- Job listings and recruitment posts
- Food / meal delivery service ads
- Real estate or rental advertisements
- News articles or editorial guides that don't describe one specific attendable event
- "Best of" roundups / listicles ("8 places to eat in Belgrade")
- Shopping guides or product launches with no public event
- Personal trainer or fitness coaching ads

Mark is_event=true for anything a person could actually attend in person.
When in doubt, prefer true — it is better to include a borderline post than
to miss a real event.

Return a JSON array with EXACTLY one object per event, in the same order as
the input, like: [{"is_event": true}, {"is_event": false}, ...]
""".strip()


class _EventDecision(BaseModel):
    is_event: bool


class _BatchDecision(BaseModel):
    decisions: list[_EventDecision]


def classify_events_batch(
    descriptions: list[str],
    *,
    model: str | None = None,
    api_key: str | None = None,
    batch_size: int = 15,
) -> list[bool]:
    """
    Classify a list of event descriptions via OpenAI.

    Returns a list[bool] of the same length — True means attendable event,
    False means promotional/editorial content that should be excluded.

    Failures in individual batches are logged and treated as True (keep the
    event) so a transient API error never silently empties the corpus.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY to use LLM event classification.")

    resolved_model = (model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = OpenAI(api_key=key)

    results: list[bool] = []

    for start in range(0, len(descriptions), batch_size):
        batch = descriptions[start : start + batch_size]
        numbered = "\n\n".join(
            f"Event {i + 1}:\n{desc[:600]}"   # cap at 600 chars to save tokens
            for i, desc in enumerate(batch)
        )
        try:
            response = client.beta.chat.completions.parse(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Classify these {len(batch)} posts.\n\n{numbered}"
                        ),
                    },
                ],
                response_format=_BatchDecision,
                temperature=0,
            )
            parsed: _BatchDecision = response.choices[0].message.parsed
            decisions = parsed.decisions

            if len(decisions) == len(batch):
                results.extend(d.is_event for d in decisions)
            elif len(decisions) > len(batch):
                # Model returned extras (common off-by-one: trailing summary item).
                # Truncate to expected length — the first N decisions are correct.
                log.debug(
                    "Classifier returned %d decisions for %d events at batch %d — truncating",
                    len(decisions), len(batch), start,
                )
                results.extend(d.is_event for d in decisions[:len(batch)])
            else:
                # Fewer than expected — pad the missing ones with True (keep event).
                log.warning(
                    "Classifier returned only %d decisions for %d events at batch %d — "
                    "padding missing with is_event=True",
                    len(decisions), len(batch), start,
                )
                results.extend(d.is_event for d in decisions)
                results.extend([True] * (len(batch) - len(decisions)))

        except Exception as exc:
            log.warning(
                "Event classifier batch %d–%d failed (%s) — defaulting to is_event=True",
                start,
                start + len(batch) - 1,
                exc,
            )
            results.extend([True] * len(batch))

    return results

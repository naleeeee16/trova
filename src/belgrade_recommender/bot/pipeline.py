"""Corpus loader + recommendation pipeline for the bot.

The corpus (6 948 events) is encoded ONCE at startup and reused for every
/find request.  All blocking calls (Gemini, encoder) run in a thread pool so
they never freeze the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from belgrade_recommender.bot.config import SCORE_FLOOR

log = logging.getLogger("bot.pipeline")

_executor = ThreadPoolExecutor(max_workers=4)

# Module-level singletons — populated by load_corpus_sync()
_events: list | None = None
_corpus_matrix = None
_encoder = None

_NORMALIZED = Path(__file__).resolve().parents[3] / "data" / "processed" / "events_normalized_test.jsonl"


def load_corpus_sync() -> None:
    """Load and encode the full event corpus.  Call once before starting the bot."""
    global _events, _corpus_matrix, _encoder


    from belgrade_recommender.retrieve.encoder import SentenceEncoder
    from belgrade_recommender.retrieve.service import load_normalized_events
    from belgrade_recommender.retrieve.text import build_embedding_text

    if not _NORMALIZED.is_file():
        raise FileNotFoundError(
            f"Normalized corpus not found: {_NORMALIZED}\n"
            "Run a test script without --skip-ingest first to generate it."
        )

    log.info("Loading corpus from %s …", _NORMALIZED)
    all_events = load_normalized_events(_NORMALIZED)
    _events = [e for e in all_events if e.include_in_index]
    log.info("%d indexable events loaded", len(_events))

    from belgrade_recommender.bot.storage import populate_events_table
    populate_events_table(all_events)

    _encoder = SentenceEncoder()
    texts = [build_embedding_text(e) for e in _events]
    t0 = time.perf_counter()
    _corpus_matrix = _encoder.encode(texts)
    log.info(
        "Corpus encoded in %.1fs — shape %s  model=%s",
        time.perf_counter() - t0,
        _corpus_matrix.shape,
        _encoder.model_name,
    )


async def recommend_for_group(
    user_texts: dict[int, str],
    lang: str = "srb",
) -> tuple[list, list, dict, list, str]:
    """
    Run the full recommendation pipeline for a group.

    Args:
        user_texts: {user_id: "combined preference text from all their messages"}
        lang: language code for generated text ("srb", "rus", "eng")

    Returns:
        (ranked, conflicts, mismatches, reasons, no_results_msg)
        ranked         — list of (NormalizedEvent, sem, min_soft, mean_soft), up to 10
        conflicts      — list of human-readable conflict strings (may be empty)
        mismatches     — {event_id: [mismatch_str, ...]} soft-preference gaps per event
        reasons        — list of short strings in the requested language, one per ranked event
        no_results_msg — LLM-generated explanation when ranked is empty; "" otherwise
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _recommend_sync, user_texts, lang)


def _recommend_sync(user_texts: dict[int, str], lang: str = "srb") -> tuple[list, list, dict, list, str]:
    from belgrade_recommender.extract.attributes import extract_event_attributes_maybe_lazy_gemini
    from belgrade_recommender.llm.openai_explain import explain_missing_tags, explain_per_event
    from belgrade_recommender.llm.openai_preferences import parse_preferences_openai
    from belgrade_recommender.rank.group_rank import rank_by_least_misery
    from belgrade_recommender.retrieve.cosine_search import cosine_top_k
    from belgrade_recommender.retrieve.hard_filters import passes_all_hard_constraints

    if not user_texts:
        log.warning("recommend called with no user inputs")
        return [], [], {}, [], [], {}, [], {}

    log.info("=== Pipeline start: %d user(s) ===", len(user_texts))
    for uid, text in user_texts.items():
        log.info("  user %d: %r", uid, text[:80])

    # 1 — Parse each user's text into structured preferences via Gemini
    #     Falls back to a permissive empty profile on rate-limit / quota errors
    #     so the bot still returns semantic results even without Gemini.
    prefs = []
    uid_to_pref: dict = {}
    summaries = []
    gemini_ok = 0
    for uid, text in user_texts.items():
        try:
            log.info("  Parsing preferences for user %d via OpenAI...", uid)
            p = parse_preferences_openai(text)
            prefs.append(p)
            uid_to_pref[uid] = p
            summaries.append(p.summary_one_line or text[:80])
            gemini_ok += 1
            log.info("  user %d OK: %s | hard=%s | soft=%s",
                uid, p.summary_one_line,
                p.hard_constraints.model_dump(exclude_defaults=True),
                p.soft_preferences.model_dump(exclude_defaults=True),
            )
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate_limit" in err.lower() or "quota" in err.lower():
                log.warning("  user %d: OpenAI rate-limited — using fallback (no hard constraints)", uid)
                fb = _fallback_prefs(text)
                prefs.append(fb)
                uid_to_pref[uid] = fb
                summaries.append(text[:80])
            else:
                log.exception("  FAILED to parse preferences for user %d", uid)
                fb = _fallback_prefs(text)
                prefs.append(fb)
                uid_to_pref[uid] = fb
                summaries.append(text[:80])

    if gemini_ok == 0:
        log.warning("OpenAI unavailable for all users — running in semantic-only fallback mode")

    log.info("Step 1 done: %d/%d parsed by OpenAI, %d via fallback",
             gemini_ok, len(user_texts), len(user_texts) - gemini_ok)

    # 1b — City backfill: if the LLM didn't set a city constraint, infer it
    #      from the raw query text using the same rules used for events.
    #      This activates the city hard filter for users who write "u Beogradu"
    #      or "in Belgrade" without the LLM capturing it in hard_constraints.city.
    from belgrade_recommender.extract.rules import detect_city_hint as _detect_city
    for p, raw_text in zip(prefs, user_texts.values()):
        if p.hard_constraints.city is None:
            inferred = _detect_city(raw_text)
            if inferred in ("belgrade", "novi_sad"):
                p.hard_constraints.city = inferred
                log.info("  city backfilled from query text: %s", inferred)

    # Niche cultural terms expanded to richer English before embedding.
    # Defined here so both the per-user query encoding (Step 2) and
    # the liked_type pre-embedding (Step 2b) can use the same map.
    _EMBED_EXPANSION: dict[str, str] = {
        "turbofolk":        "turbofolk Serbian folk pop music party dance",
        "folk_traditional": "traditional folk music Serbian narodni kafana",
        "kafana":           "kafana Serbian tavern folk music traditional drinks",
        "rakija":           "rakija Serbian traditional spirits folk tavern",
        "splav":            "splav river raft party club Belgrade nightlife",
        "dark_techno":      "dark techno industrial underground club night",
        "dark_ambient":     "dark ambient industrial gothic experimental music",
    }

    # 2 — Semantic search: encode each user's query independently then average.
    # Averaging per-user vectors (rather than concatenating all text) gives every member
    # equal weight regardless of how long or short their query is.  A single short
    # "splav na reci" query would otherwise be drowned out by a longer food query.
    import numpy as _np2
    log.info("Step 2: encoding query and searching corpus...")
    per_user_vecs = []
    for uid, utext in user_texts.items():
        p_for_uid = uid_to_pref.get(uid)
        lt_terms = (p_for_uid.soft_preferences.liked_types or []) if p_for_uid else []
        liked_str = " ".join(
            _EMBED_EXPANSION.get(t, t.replace("_", " ")) for t in lt_terms
        )
        per_query = (utext + ("\n\n" + liked_str if liked_str else "")).strip()
        per_user_vecs.append(_encoder.encode([per_query])[0])
    query_vec = _np2.mean(per_user_vecs, axis=0)
    pool_hits = cosine_top_k(_corpus_matrix, query_vec, k=80)
    log.info("  Semantic pool: %d candidates", len(pool_hits))

    # Keyword fallback: if the best cosine score is below the threshold the
    # embedding model essentially returned noise.  Augment the candidate pool
    # with full-text keyword matches so the pipeline always has something to work with.
    _SEM_FALLBACK_THRESHOLD = 0.30
    top_sem_score = pool_hits[0][1] if pool_hits else 0.0
    if top_sem_score < _SEM_FALLBACK_THRESHOLD:
        log.warning(
            "Top semantic score %.3f < %.2f — triggering keyword fallback",
            top_sem_score, _SEM_FALLBACK_THRESHOLD,
        )
        from belgrade_recommender.retrieve.keyword_search import keyword_top_k
        combined_query = " ".join(user_texts.values())
        kw_hits = keyword_top_k(_events, combined_query, k=80)
        existing_indices = {idx for idx, _ in pool_hits}
        added = 0
        for idx, kw_score in kw_hits:
            if idx not in existing_indices:
                pool_hits.append((idx, kw_score))
                existing_indices.add(idx)
                added += 1
        log.info(
            "  Keyword fallback: %d new candidates added (total pool: %d)",
            added, len(pool_hits),
        )

    # 2b — Pre-embed each user's liked_types once so the scorer can use cosine
    #      similarity instead of brittle substring matching.

    liked_vecs_list = []
    for p in prefs:
        lt_terms = p.soft_preferences.liked_types
        if lt_terms:
            phrases = [
                _EMBED_EXPANSION.get(t, t.replace("_", " "))
                for t in lt_terms
            ]
            liked_vecs_list.append(_encoder.encode(phrases))
        else:
            liked_vecs_list.append(None)

    # 3 — Hard filter: keep only events every member can attend
    #     Also store attrs and event vectors so soft_score can use cosine matching.
    hards = [p.hard_constraints for p in prefs]
    candidates = []
    attrs_map: dict[str, object] = {}
    event_vecs: dict[str, object] = {}
    for idx, score in pool_hits:
        event = _events[idx]
        attrs = extract_event_attributes_maybe_lazy_gemini(event)
        if passes_all_hard_constraints(hards, event, attrs):
            candidates.append((event, score))
            attrs_map[event.event_id] = attrs
            event_vecs[event.event_id] = _corpus_matrix[idx]
        if len(candidates) >= 20:
            break

    log.info("Step 3: hard filter → %d candidates passed", len(candidates))

    if not candidates:
        log.warning("No candidates survived hard filtering — check constraints")
        return [], [], {}, [], prefs, attrs_map, liked_vecs_list, event_vecs

    # 3b — Detect conflicting preferences so the bot can warn the group upfront
    conflicts = _detect_preference_conflicts(prefs, lang=lang)
    if conflicts:
        log.info("Preference conflicts detected: %s", conflicts)

    # 4 — Least-misery ranking + score floor
    # Results below SCORE_FLOOR min_soft are essentially random noise from the
    # embedding space and are never useful to show the user.
    ranked = [
        r for r in rank_by_least_misery(
            candidates, prefs,
            liked_vecs_list=liked_vecs_list,
            event_vecs=event_vecs,
        )[:10]
        if r[2] >= SCORE_FLOOR
    ]
    if not ranked:
        log.warning(
            "All %d candidates below score floor %.2f — returning empty",
            len(candidates), SCORE_FLOOR,
        )
        return [], conflicts, {}, [], prefs, attrs_map, liked_vecs_list, event_vecs
    log.info("Step 4: ranked %d events — top min_score=%.3f", len(ranked), ranked[0][2])

    # 4a — Detailed per-event log so you can see exactly what was recommended and why
    import numpy as _np
    _NONE = "(none)"
    _SEP = "-" * 72
    log.info("")
    log.info("=" * 72)
    log.info("RANKED RESULTS — %d events", len(ranked))
    log.info("=" * 72)
    for rank_i, (ev, sem, mn, mean) in enumerate(ranked, 1):
        attrs = attrs_map.get(ev.event_id)
        ev_vec = event_vecs.get(ev.event_id)
        log.info("")
        log.info("%s", _SEP)
        log.info("#%d  %s", rank_i, (ev.event_description_resolved or "")[:100])
        log.info("    link       : %s", ev.link or _NONE)
        log.info("    SCORES     : sem=%.4f  min_soft=%.4f  mean_soft=%.4f  [group=%d%%]",
                 sem, mn, mean, int(mn * 100))
        log.info("    raw tags   : %s", " | ".join(ev.tags or []) or _NONE)
        if attrs:
            log.info("    type_hints : %s", " | ".join(getattr(attrs, "type_hints", [])) or _NONE)
            log.info("    noise      : %s", getattr(attrs, "noise_level_hint", "?"))
            log.info("    outdoor    : %s", getattr(attrs, "outdoor_hint", "?"))
            log.info("    city       : %s", getattr(attrs, "city_hint", "?"))
            log.info("    free       : %s  paid=%s  rsd=%s  eur=%s",
                     getattr(attrs, "price_free_signal", "?"),
                     getattr(attrs, "price_paid_signal", "?"),
                     getattr(attrs, "price_amount_rsd", None) or "-",
                     getattr(attrs, "price_amount_eur", None) or "-")
        # Per-user cosine breakdown
        if ev_vec is not None:
            ev_norm = float(_np.linalg.norm(ev_vec)) + 1e-9
            for ui, (p, lv) in enumerate(zip(prefs, liked_vecs_list)):
                liked_tokens = p.soft_preferences.liked_types
                if lv is not None and len(lv) > 0 and liked_tokens:
                    lt_norms = _np.linalg.norm(lv, axis=1) + 1e-9
                    cosines = (lv @ ev_vec) / (lt_norms * ev_norm)
                    pairs = "  ".join(
                        f"{tok}={'✓' if c>=0.35 else '✗'}{c:.2f}"
                        for tok, c in zip(liked_tokens, cosines)
                    )
                    log.info("    [user %d] noise=%-12s liked: %s", ui + 1,
                             p.soft_preferences.noise_tolerance, pairs)
                else:
                    log.info("    [user %d] noise=%-12s (no liked_types)", ui + 1,
                             p.soft_preferences.noise_tolerance)
    log.info("")
    log.info("=" * 72)
    log.info("")

    # 4b — Compute soft-preference mismatches per event
    mismatches: dict[str, list[str]] = {}
    for ev, sem, mn, mean in ranked:
        attrs = attrs_map.get(ev.event_id)
        mismatches[ev.event_id] = _compute_mismatches(prefs, attrs, lang=lang)

    # 5 — Per-event explanations via a single OpenAI call
    reasons: list[str] = []
    try:
        log.info("Step 5: generating per-event explanations (lang=%s)...", lang)
        reasons = explain_per_event(
            preferences_context="\n".join(f"- {s}" for s in summaries),
            ranked=ranked,
            lang=lang,
        )
        log.info("  Reasons generated for %d events", len(reasons))
    except Exception:
        log.warning("Per-event explanation failed — continuing without it")
        reasons = [""] * len(ranked)

    log.info("=== Pipeline done: returning %d results ===", len(ranked))
    # `explanation` slot (2nd element) reused for conflict warnings — list of
    # Serbian strings the bot surfaces before the result list.
    return ranked, conflicts, mismatches, reasons, prefs, attrs_map, liked_vecs_list, event_vecs


_CONFLICT_CATEGORIES = {
    "srb": "suprotstavljene kategorije ({terms})",
    "rus": "противоречивые категории ({terms})",
    "eng": "conflicting categories ({terms})",
}

_CONFLICT_NOISE = {
    "srb": "neko želi glasnu atmosferu, neko tiho okruženje",
    "rus": "кто-то хочет громкую атмосферу, кто-то тихую",
    "eng": "someone wants a loud atmosphere, someone wants a quiet one",
}


def _detect_preference_conflicts(prefs: list, lang: str = "srb") -> list[str]:
    """
    Return a list of human-readable conflict strings (in the requested language) when
    one member's liked_types appear in another member's forbidden_event_categories,
    or when noise tolerances are directly opposed (high vs low).
    """
    conflicts: list[str] = []

    cat_tmpl = _CONFLICT_CATEGORIES.get(lang, _CONFLICT_CATEGORIES["srb"])
    noise_msg = _CONFLICT_NOISE.get(lang, _CONFLICT_NOISE["srb"])

    for i in range(len(prefs)):
        for j in range(i + 1, len(prefs)):
            p1, p2 = prefs[i], prefs[j]

            liked1 = {t.lower() for t in p1.soft_preferences.liked_types}
            forbidden2 = {t.lower() for t in p2.hard_constraints.forbidden_event_categories}
            liked2 = {t.lower() for t in p2.soft_preferences.liked_types}
            forbidden1 = {t.lower() for t in p1.hard_constraints.forbidden_event_categories}

            clashing = (liked1 & forbidden2) | (liked2 & forbidden1)
            if clashing:
                terms = ", ".join(sorted(clashing)[:3])
                conflicts.append(cat_tmpl.format(terms=terms))

        # Noise tolerance clash: loud vs quiet
        n1 = prefs[i].soft_preferences.noise_tolerance
        for j in range(i + 1, len(prefs)):
            n2 = prefs[j].soft_preferences.noise_tolerance
            if {n1, n2} == {"high", "low"}:
                conflicts.append(noise_msg)

    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    unique: list[str] = []
    for c in conflicts:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


_MM_LOUD = {
    "srb": "glasno (neko želi tiho)",
    "rus": "громко (кто-то хочет тишину)",
    "eng": "loud (someone wants quiet)",
}

_MM_QUIET = {
    "srb": "tiho (neko želi glasnu muziku)",
    "rus": "тихо (кто-то хочет громкую музыку)",
    "eng": "quiet (someone wants loud music)",
}

_MM_TYPE = {
    "srb": "tip ne odgovara ({missing})",
    "rus": "тип не подходит ({missing})",
    "eng": "type doesn't match ({missing})",
}


def _compute_mismatches(prefs: list, attrs, lang: str = "srb") -> list[str]:
    """Return human-readable soft-preference gaps for one event."""
    if attrs is None:
        return []

    issues = []

    loud_msg  = _MM_LOUD.get(lang, _MM_LOUD["srb"])
    quiet_msg = _MM_QUIET.get(lang, _MM_QUIET["srb"])
    type_tmpl = _MM_TYPE.get(lang, _MM_TYPE["srb"])

    # Noise mismatch
    noise_hint = getattr(attrs, "noise_level_hint", "unknown")
    if noise_hint != "unknown":
        wants_quiet = any(
            getattr(p.soft_preferences, "noise_tolerance", "unknown") == "low"
            for p in prefs
        )
        wants_loud = any(
            getattr(p.soft_preferences, "noise_tolerance", "unknown") == "high"
            for p in prefs
        )
        if wants_quiet and noise_hint == "high":
            issues.append(loud_msg)
        if wants_loud and noise_hint == "low":
            issues.append(quiet_msg)

    # Liked-types mismatch — someone has specific liked types but event doesn't match any
    event_types = set(getattr(attrs, "type_hints", []))
    for p in prefs:
        liked = set(getattr(p.soft_preferences, "liked_types", []))
        if liked and event_types and not liked.intersection(event_types):
            missing = ", ".join(sorted(liked)[:2])
            issues.append(type_tmpl.format(missing=missing))
            break  # report once for the group

    return issues


def _fallback_prefs(text: str):
    """Return a permissive preference profile when Gemini is unavailable.
    No hard constraints — just uses the raw text for semantic search."""
    from belgrade_recommender.schemas.parsed_preferences import (
        ParsedHardConstraints,
        ParsedSoftPreferences,
        ParsedUserPreferences,
    )
    return ParsedUserPreferences(
        summary_one_line=text[:80],
        hard_constraints=ParsedHardConstraints(),
        soft_preferences=ParsedSoftPreferences(),
    )

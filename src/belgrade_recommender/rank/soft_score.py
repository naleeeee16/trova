"""Soft preference signals for scoring"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.schemas.event_attributes import EventAttributes
from belgrade_recommender.schemas.parsed_preferences import ParsedUserPreferences

if TYPE_CHECKING:
    pass

# Cosine threshold above which a liked_type embedding is considered a match
# against the event embedding. Calibrated for paraphrase-multilingual-MiniLM-L12-v2.
# 0.35 = "related concept", 0.50 = "clearly the same topic".
_SEMANTIC_MATCH_THRESHOLD: float = 0.35

# Cosine above this is a "strong" match (clearly the same topic, not just adjacent).
_SEMANTIC_STRONG_THRESHOLD: float = 0.50

# When ALL liked_type cosines are below this, the event is a confirmed mismatch.
_SEMANTIC_MISMATCH_CAP: float = 0.20

# Semantic similarity below this margin gets quadratically compressed.
_SQUASH_MARGIN: float = 0.40

# Penalty for confirmed type mismatch (liked_types present but nothing relevant).
_MISMATCH_PENALTY: float = 0.10

# Bonus per strong match (cosine >= 0.50) vs weak match (0.35–0.50).
_STRONG_MATCH_BONUS: float = 0.08
_WEAK_MATCH_BONUS: float = 0.03

# Minimal synonym map used ONLY in the substring fallback path (test scripts
# that don't provide embedding vectors). The bot always uses the semantic path.
_LIKED_TYPE_SYNONYMS: dict[str, str] = {
    "etno": "folk_traditional",
    "narodni": "folk_traditional",
    "turbofolk": "folk_traditional",
    "swing": "jazz",
    "rnb": "soul",
    "cocktail": "cocktail_bar",
    "rooftop": "rooftop_bar",
    "dorucak": "breakfast",
    "doručak": "breakfast",
    "rucak": "lunch",
    "vecera": "dinner",
}


def soft_adjusted_score(
    prefs: ParsedUserPreferences,
    event: NormalizedEvent,
    attrs: EventAttributes,
    semantic_similarity: float,
    liked_vecs: "np.ndarray | None" = None,
    event_vec: "np.ndarray | None" = None,
) -> float:
    """
    Combine semantic similarity with soft bonuses/penalties for one (user, event) pair.

    Score pipeline:
      1. Soft-squash base if semantic signal is weak (< _SQUASH_MARGIN)
      2. Liked-type bonus (tiered, cap +0.28):
           strong match (cosine >= 0.50): +0.08 each
           weak match  (cosine 0.35–0.50): +0.03 each
         Partial mismatch penalty: proportional to fraction of liked_types that missed.
         Full mismatch penalty: all liked_types far below _SEMANTIC_MISMATCH_CAP.
      3. Noise mismatch penalty
    """
    base = max(0.0, min(1.0, float(semantic_similarity)))

    # 1 — Soft squash
    if base < _SQUASH_MARGIN:
        base = (base ** 2) / _SQUASH_MARGIN

    # 2 — Liked-type bonus
    liked_types = prefs.soft_preferences.liked_types
    matched_count = 0
    penalty = 0.0

    bonus = 0.0
    if liked_types:
        if liked_vecs is not None and event_vec is not None and len(liked_vecs) > 0:
            # Semantic path: cosine between each liked_type embedding and the event embedding.
            ev_norm = float(np.linalg.norm(event_vec)) + 1e-9
            lt_norms = np.linalg.norm(liked_vecs, axis=1) + 1e-9
            cosines = (liked_vecs @ event_vec) / (lt_norms * ev_norm)

            # Tiered bonus: strong match (>=0.50) earns more than a weak/adjacent match (0.35–0.50).
            # This prevents semantically adjacent categories (gin ≈ craft_beer) from scoring
            # as well as exact matches, without raising the threshold and losing jazz≈swing.
            for c in cosines:
                if c >= _SEMANTIC_STRONG_THRESHOLD:
                    bonus += _STRONG_MATCH_BONUS
                    matched_count += 1
                elif c >= _SEMANTIC_MATCH_THRESHOLD:
                    bonus += _WEAK_MATCH_BONUS
                    matched_count += 1

            # Partial mismatch: user wanted N types but only M matched → proportional penalty.
            # Prevents one matching preference from hiding that another was completely missed.
            missed = len(liked_types) - matched_count
            if missed > 0:
                penalty += (missed / len(liked_types)) * _MISMATCH_PENALTY

            # Full mismatch: all liked_types are completely unrelated to the event.
            if float(np.max(cosines)) < _SEMANTIC_MISMATCH_CAP:
                penalty += _MISMATCH_PENALTY
        else:
            # Fallback: substring matching against extracted type_hints + tags
            liked = [
                _LIKED_TYPE_SYNONYMS.get(t.lower(), t.lower())
                for t in liked_types
            ]
            tag_hint = {t.lower() for t in event.tags} | {h.lower() for h in attrs.type_hints}
            for lt in liked:
                if any(lt in x or x in lt for x in tag_hint):
                    matched_count += 1
                    bonus += _STRONG_MATCH_BONUS
            if liked and attrs.type_hints and matched_count == 0:
                penalty += _MISMATCH_PENALTY

    bonus = min(bonus, 0.28)

    # 3 — Noise mismatch penalty
    noise = prefs.soft_preferences.noise_tolerance
    event_noise = attrs.noise_level_hint
    effective_event_noise = event_noise if event_noise != "unknown" else "medium"

    if noise in ("low", "unknown") and event_noise == "high":
        penalty += 0.12
    if noise == "low" and effective_event_noise == "medium":
        penalty += 0.04
    if noise == "medium" and event_noise == "high":
        penalty += 0.06

    return max(0.0, min(1.0, base + bonus - penalty))

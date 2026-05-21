"""Full-text keyword search fallback.

Used when the top cosine similarity score falls below the configured threshold,
meaning the embedding model failed to find semantically close events.  Scans
event descriptions and tags for literal query token matches, producing a
candidate pool that the normal hard-filter + ranking pipeline then processes.
"""

from __future__ import annotations

import re

from belgrade_recommender.ingest.models import NormalizedEvent

# Tokens shorter than this are skipped (too noisy / likely stopwords).
_MIN_TOKEN_LEN = 3

_STOPWORDS = frozenset({
    # English
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "be", "it", "its", "this", "that", "with",
    "from", "by", "as", "not", "but", "my", "our", "we", "you", "your",
    # Serbian / Bosnian / Croatian
    "u", "na", "za", "da", "ne", "ni", "se", "sa", "je", "su", "bi",
    "ce", "će", "li", "al", "ali", "ili", "kao", "taj", "ta", "to",
    "ovo", "ovo", "bez", "jos", "već", "vec", "sve", "svi", "sva",
    # Russian (transliterated common)
    "в", "на", "и", "с", "не", "но", "по", "из", "за", "от",
})

# Assigned as the ``sem`` field for keyword-matched events so downstream
# soft scoring can still apply liked_type bonuses.  Kept below the
# semantic fallback threshold (0.30) to signal non-semantic origin.
_KW_SEM_SCORE: float = 0.25


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word chars, remove stopwords and short tokens."""
    raw = re.findall(r"\w+", text.lower())
    return [t for t in raw if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS]


def keyword_top_k(
    events: list[NormalizedEvent],
    query: str,
    k: int = 80,
    sem_score: float = _KW_SEM_SCORE,
) -> list[tuple[int, float]]:
    """Full-text keyword search over event descriptions and tags.

    Args:
        events:     Indexable event list (same order as the corpus matrix).
        query:      Raw combined user preference text.
        k:          Maximum number of results to return.
        sem_score:  The ``sem`` value injected for keyword-matched events so they
                    can proceed through soft scoring and ranking.

    Returns:
        List of ``(event_index, sem_score)`` tuples, sorted by descending
        number of matched query tokens (best keyword overlap first).
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored: list[tuple[int, int]] = []
    for idx, event in enumerate(events):
        desc = (event.event_description_resolved or "").lower()
        tags = " ".join(event.tags or []).lower()
        combined = desc + " " + tags
        hit_count = sum(1 for t in query_tokens if t in combined)
        if hit_count > 0:
            scored.append((idx, hit_count))

    scored.sort(key=lambda x: -x[1])
    return [(idx, sem_score) for idx, _ in scored[:k]]

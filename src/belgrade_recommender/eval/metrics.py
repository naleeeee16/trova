"""Offline relevance metrics (labels are provided by humans, not inferred)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(relevant: set[str], ranked_ids: Sequence[str], k: int) -> float:
    """Fraction of the top-k slots that hit a relevant id.

    Denominator is min(k, len(ranked_ids)) so short result lists don't inflate the score.
    """
    if k <= 0:
        return 0.0
    k_eff = min(k, len(ranked_ids))
    if k_eff == 0:
        return 0.0
    top = ranked_ids[:k_eff]
    hits = sum(1 for eid in top if eid in relevant)
    return hits / k_eff


def recall_at_k(relevant: set[str], ranked_ids: Sequence[str], k: int) -> float:
    """Fraction of all relevant ids that appear in the top-k list."""
    if not relevant:
        return 0.0
    top = set(ranked_ids[: min(k, len(ranked_ids))])
    return len(relevant & top) / len(relevant)


def reciprocal_rank(relevant: set[str], ranked_ids: Sequence[str]) -> float:
    """1 / rank of the first relevant hit (1-based); 0 if none."""
    for i, eid in enumerate(ranked_ids, start=1):
        if eid in relevant:
            return 1.0 / float(i)
    return 0.0


def ndcg_at_k(relevant: set[str], ranked_ids: Sequence[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance).

    DCG@k  = sum(1 / log2(rank+1)) for each relevant hit in top-k
    IDCG@k = DCG of the ideal ranking (all relevant items first)
    NDCG   = DCG / IDCG
    """
    if k <= 0 or not relevant:
        return 0.0
    top = ranked_ids[: min(k, len(ranked_ids))]
    dcg = sum(
        1.0 / math.log2(i + 2)  # i is 0-based → rank = i+1 → log2(rank+1)
        for i, eid in enumerate(top)
        if eid in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def jains_fairness_index(scores: Sequence[float]) -> float:
    """Jain's Fairness Index over per-member scores.

    Returns 1.0 when all members get equal scores (perfectly fair);
    approaches 1/n when one member dominates.  Measures how evenly the
    recommendation quality is distributed across group members.
    """
    if not scores:
        return 0.0
    n = len(scores)
    sum_sq = sum(s * s for s in scores)
    if sum_sq < 1e-12:
        return 0.0
    return (sum(scores) ** 2) / (n * sum_sq)


def least_misery_gap(scores: Sequence[float]) -> float:
    """Max–min spread of per-member soft scores.

    Lower = fairer group recommendation (the worst-off member is not far
    behind the best-served one).  0.0 means all members score equally.
    """
    if not scores:
        return 0.0
    return max(scores) - min(scores)

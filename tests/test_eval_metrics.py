import math

from belgrade_recommender.eval.metrics import (
    jains_fairness_index,
    least_misery_gap,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k() -> None:
    rel = {"a", "b"}
    ranked = ["x", "a", "b", "c"]
    assert precision_at_k(rel, ranked, k=3) == 2 / 3
    assert precision_at_k(rel, ranked, k=10) == 2 / 4


def test_recall_at_k() -> None:
    rel = {"a", "b", "z"}
    ranked = ["a", "x", "b"]
    assert recall_at_k(rel, ranked, k=3) == 2 / 3


def test_reciprocal_rank() -> None:
    rel = {"b"}
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(rel, ranked) == 0.5


def test_empty_ranked() -> None:
    assert precision_at_k({"a"}, [], k=5) == 0.0
    assert recall_at_k({"a"}, [], k=5) == 0.0
    assert reciprocal_rank({"a"}, []) == 0.0


def test_ndcg_at_k_perfect() -> None:
    # Relevant item at rank 1 → NDCG = 1.0
    rel = {"a"}
    ranked = ["a", "b", "c"]
    assert ndcg_at_k(rel, ranked, k=3) == 1.0


def test_ndcg_at_k_partial() -> None:
    # Two relevant items; first at rank 2, second at rank 3
    rel = {"b", "c"}
    ranked = ["a", "b", "c"]
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)   # ranks 2 and 3
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # ideal: ranks 1 and 2
    assert abs(ndcg_at_k(rel, ranked, k=3) - dcg / idcg) < 1e-9


def test_ndcg_at_k_miss() -> None:
    assert ndcg_at_k({"z"}, ["a", "b", "c"], k=3) == 0.0


def test_ndcg_at_k_empty() -> None:
    assert ndcg_at_k(set(), ["a"], k=5) == 0.0
    assert ndcg_at_k({"a"}, [], k=5) == 0.0


def test_jains_fairness_index_equal() -> None:
    # All equal scores → fairness = 1.0
    assert abs(jains_fairness_index([0.5, 0.5, 0.5]) - 1.0) < 1e-9


def test_jains_fairness_index_unequal() -> None:
    # One member gets everything → approaches 1/n
    scores = [1.0, 0.0, 0.0]
    assert jains_fairness_index(scores) < 0.4


def test_jains_fairness_index_empty() -> None:
    assert jains_fairness_index([]) == 0.0


def test_least_misery_gap() -> None:
    assert abs(least_misery_gap([0.8, 0.5, 0.6]) - 0.3) < 1e-9
    assert least_misery_gap([0.7, 0.7]) == 0.0
    assert least_misery_gap([]) == 0.0

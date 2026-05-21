import pytest

np = pytest.importorskip("numpy")

from belgrade_recommender.retrieve.cosine_search import cosine_top_k


def test_cosine_top_k_orders_by_dot_product() -> None:
    # Three normalized rows in R^2; query closest to row 1
    corpus = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([0.0, 1.0], dtype=np.float32)
    top = cosine_top_k(corpus, query, k=2)
    assert top[0][0] == 1
    assert top[1][0] in (0, 2)

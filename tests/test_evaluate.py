"""CP2.4 checks for Recall@K and NDCG@K on hand-computed toy cases."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from models.evaluate import ranking_metrics


def test_perfect_ranking_scores_one():
    recs = np.array([[3, 7, 1]])
    metrics = ranking_metrics(recs, np.array([0]), {0: np.array([3, 7])}, k=3)
    assert metrics["recall@3"] == 1.0
    assert math.isclose(metrics["ndcg@3"], 1.0)


def test_single_hit_at_rank_two():
    recs = np.array([[5, 3, 1]])
    metrics = ranking_metrics(recs, np.array([0]), {0: np.array([3, 9])}, k=3)
    assert metrics["recall@3"] == 0.5
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(metrics["ndcg@3"], expected)


def test_users_without_targets_are_skipped():
    recs = np.array([[1, 2], [3, 4]])
    metrics = ranking_metrics(recs, np.array([0, 1]), {1: np.array([4])}, k=2)
    assert metrics["num_users"] == 1
    assert metrics["recall@2"] == 1.0

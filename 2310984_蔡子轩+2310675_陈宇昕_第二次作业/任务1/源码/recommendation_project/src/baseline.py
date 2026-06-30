"""Baseline predictor with global mean, user bias and item bias."""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from utils import Prediction, TestPair, TrainRecord


class BaselineModel:
    """
    Simple baseline recommender.

    Prediction rule:
        r_hat(u, i) = global_mean + user_bias[u] + item_bias[i]

    The user/item biases are learned by regularized alternating updates.
    """

    def __init__(
        self,
        reg_user: float = 10.0,
        reg_item: float = 15.0,
        n_iters: int = 15,
        seed: int = 2026,
    ) -> None:
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.n_iters = n_iters
        self.seed = seed

        self.global_mean: float = 0.0
        self.user_bias: Dict[int, float] = {}
        self.item_bias: Dict[int, float] = {}
        self.min_rating: float = 0.0
        self.max_rating: float = 0.0

    def fit(self, train_data: Sequence[TrainRecord]) -> "BaselineModel":
        if not train_data:
            raise ValueError("train_data must not be empty")

        user_ratings: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        item_ratings: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        ratings = []

        for user_id, item_id, rating in train_data:
            user_ratings[user_id].append((item_id, rating))
            item_ratings[item_id].append((user_id, rating))
            ratings.append(rating)

        self.global_mean = sum(ratings) / len(ratings)
        self.min_rating = min(ratings)
        self.max_rating = max(ratings)
        self.user_bias = {user_id: 0.0 for user_id in user_ratings}
        self.item_bias = {item_id: 0.0 for item_id in item_ratings}

        for _ in range(self.n_iters):
            for user_id, records in user_ratings.items():
                residual_sum = sum(
                    rating - self.global_mean - self.item_bias.get(item_id, 0.0)
                    for item_id, rating in records
                )
                self.user_bias[user_id] = residual_sum / (self.reg_user + len(records))

            for item_id, records in item_ratings.items():
                residual_sum = sum(
                    rating - self.global_mean - self.user_bias.get(user_id, 0.0)
                    for user_id, rating in records
                )
                self.item_bias[item_id] = residual_sum / (self.reg_item + len(records))

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        return (
            self.global_mean
            + self.user_bias.get(user_id, 0.0)
            + self.item_bias.get(item_id, 0.0)
        )

    def batch_predict(self, test_pairs: Sequence[TestPair]) -> List[Prediction]:
        return [
            (user_id, item_id, self.predict(user_id, item_id))
            for user_id, item_id in test_pairs
        ]

    def approximate_size_bytes(self) -> int:
        size = sys.getsizeof(self)
        size += sys.getsizeof(self.user_bias)
        size += sys.getsizeof(self.item_bias)
        for key, value in self.user_bias.items():
            size += sys.getsizeof(key) + sys.getsizeof(value)
        for key, value in self.item_bias.items():
            size += sys.getsizeof(key) + sys.getsizeof(value)
        return size


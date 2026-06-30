"""Matrix factorization with biases trained by SGD."""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence

import numpy as np

from baseline import BaselineModel
from utils import Prediction, TestPair, TrainRecord


class MatrixFactorizationModel:
    """
    Biased matrix factorization.

    Prediction rule:
        r_hat(u, i) = global_mean + bu[u] + bi[i] + P[u] dot Q[i]
    """

    def __init__(
        self,
        n_factors: int = 20,
        n_epochs: int = 25,
        learning_rate: float = 0.01,
        reg: float = 0.02,
        reg_bias: float = 0.02,
        init_std: float = 0.01,
        init_with_baseline: bool = True,
        freeze_bias: bool = False,
        clip_during_training: bool = False,
        baseline_reg_user: float = 10.0,
        baseline_reg_item: float = 15.0,
        baseline_n_iters: int = 15,
        seed: int = 2026,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.reg_bias = reg_bias
        self.init_std = init_std
        self.init_with_baseline = init_with_baseline
        self.freeze_bias = freeze_bias
        self.clip_during_training = clip_during_training
        self.baseline_reg_user = baseline_reg_user
        self.baseline_reg_item = baseline_reg_item
        self.baseline_n_iters = baseline_n_iters
        self.seed = seed

        self.global_mean: float = 0.0
        self.min_rating: float = 0.0
        self.max_rating: float = 0.0

        self.user_to_idx: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, train_data: Sequence[TrainRecord]) -> "MatrixFactorizationModel":
        if not train_data:
            raise ValueError("train_data must not be empty")

        user_ids = sorted({user_id for user_id, _, _ in train_data})
        item_ids = sorted({item_id for _, item_id, _ in train_data})
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

        ratings = np.array([rating for _, _, rating in train_data], dtype=np.float32)
        self.global_mean = float(ratings.mean())
        self.min_rating = float(ratings.min())
        self.max_rating = float(ratings.max())

        user_indices = np.array(
            [self.user_to_idx[user_id] for user_id, _, _ in train_data],
            dtype=np.int32,
        )
        item_indices = np.array(
            [self.item_to_idx[item_id] for _, item_id, _ in train_data],
            dtype=np.int32,
        )

        rng = np.random.default_rng(self.seed)
        n_users = len(user_ids)
        n_items = len(item_ids)

        if self.init_with_baseline:
            baseline = BaselineModel(
                reg_user=self.baseline_reg_user,
                reg_item=self.baseline_reg_item,
                n_iters=self.baseline_n_iters,
                seed=self.seed,
            )
            baseline.fit(train_data)
            self.global_mean = baseline.global_mean
            self.user_bias = np.array(
                [baseline.user_bias.get(user_id, 0.0) for user_id in user_ids],
                dtype=np.float32,
            )
            self.item_bias = np.array(
                [baseline.item_bias.get(item_id, 0.0) for item_id in item_ids],
                dtype=np.float32,
            )
        else:
            self.user_bias = np.zeros(n_users, dtype=np.float32)
            self.item_bias = np.zeros(n_items, dtype=np.float32)

        self.user_factors = rng.normal(
            loc=0.0,
            scale=self.init_std,
            size=(n_users, self.n_factors),
        ).astype(np.float32)
        self.item_factors = rng.normal(
            loc=0.0,
            scale=self.init_std,
            size=(n_items, self.n_factors),
        ).astype(np.float32)

        order = np.arange(len(train_data), dtype=np.int32)
        learning_rate = self.learning_rate

        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for idx in order:
                user_idx = user_indices[idx]
                item_idx = item_indices[idx]
                rating = ratings[idx]

                pred = (
                    self.global_mean
                    + self.user_bias[user_idx]
                    + self.item_bias[item_idx]
                    + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                )
                if self.clip_during_training:
                    pred = np.float32(np.clip(pred, self.min_rating, self.max_rating))
                error = rating - pred

                user_vector = self.user_factors[user_idx].copy()
                item_vector = self.item_factors[item_idx].copy()

                if not self.freeze_bias:
                    self.user_bias[user_idx] += learning_rate * (
                        error - self.reg_bias * self.user_bias[user_idx]
                    )
                    self.item_bias[item_idx] += learning_rate * (
                        error - self.reg_bias * self.item_bias[item_idx]
                    )
                self.user_factors[user_idx] += learning_rate * (
                    error * item_vector - self.reg * user_vector
                )
                self.item_factors[item_idx] += learning_rate * (
                    error * user_vector - self.reg * item_vector
                )

            learning_rate *= 0.95

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        score = self.global_mean
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(item_id)

        if user_idx is not None and self.user_bias is not None:
            score += float(self.user_bias[user_idx])
        if item_idx is not None and self.item_bias is not None:
            score += float(self.item_bias[item_idx])
        if (
            user_idx is not None
            and item_idx is not None
            and self.user_factors is not None
            and self.item_factors is not None
        ):
            score += float(np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        return score

    def batch_predict(self, test_pairs: Sequence[TestPair]) -> List[Prediction]:
        return [
            (user_id, item_id, self.predict(user_id, item_id))
            for user_id, item_id in test_pairs
        ]

    def approximate_size_bytes(self) -> int:
        size = sys.getsizeof(self)
        size += sys.getsizeof(self.user_to_idx) + sys.getsizeof(self.item_to_idx)
        for key, value in self.user_to_idx.items():
            size += sys.getsizeof(key) + sys.getsizeof(value)
        for key, value in self.item_to_idx.items():
            size += sys.getsizeof(key) + sys.getsizeof(value)

        if self.user_bias is not None:
            size += self.user_bias.nbytes
        if self.item_bias is not None:
            size += self.item_bias.nbytes
        if self.user_factors is not None:
            size += self.user_factors.nbytes
        if self.item_factors is not None:
            size += self.item_factors.nbytes
        return size

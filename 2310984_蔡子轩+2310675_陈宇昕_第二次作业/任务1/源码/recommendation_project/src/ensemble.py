"""Linear ensemble of recommendation models."""

from __future__ import annotations

import sys
from typing import List, Sequence

from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from utils import Prediction, TestPair, TrainRecord


class EnsembleModel:
    """
    Weighted average of ItemCF and Matrix Factorization predictions.

    Prediction rule:
        r_hat = itemcf_weight * itemcf_pred + (1 - itemcf_weight) * mf_pred
    """

    def __init__(
        self,
        itemcf_weight: float = 0.55,
        seed: int = 2026,
    ) -> None:
        if not 0.0 <= itemcf_weight <= 1.0:
            raise ValueError("itemcf_weight must be in [0, 1].")

        self.itemcf_weight = itemcf_weight
        self.seed = seed
        self.itemcf = ItemCFModel(
            top_k=40,
            similarity_top_n=200,
            shrinkage=50.0,
            min_common=2,
            block_size=256,
            seed=seed,
        )
        self.matrix_factorization = MatrixFactorizationModel(
            n_factors=4,
            n_epochs=12,
            learning_rate=0.002,
            reg=0.2,
            reg_bias=0.05,
            init_std=0.01,
            init_with_baseline=True,
            freeze_bias=True,
            seed=seed,
        )
        self.min_rating: float = 0.0
        self.max_rating: float = 0.0

    def fit(self, train_data: Sequence[TrainRecord]) -> "EnsembleModel":
        self.itemcf.fit(train_data)
        self.matrix_factorization.fit(train_data)
        self.min_rating = self.itemcf.min_rating
        self.max_rating = self.itemcf.max_rating
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        itemcf_score = self.itemcf.predict(user_id, item_id)
        mf_score = self.matrix_factorization.predict(user_id, item_id)
        return (
            self.itemcf_weight * itemcf_score
            + (1.0 - self.itemcf_weight) * mf_score
        )

    def batch_predict(self, test_pairs: Sequence[TestPair]) -> List[Prediction]:
        return [
            (user_id, item_id, self.predict(user_id, item_id))
            for user_id, item_id in test_pairs
        ]

    def approximate_size_bytes(self) -> int:
        size = sys.getsizeof(self)
        size += self.itemcf.approximate_size_bytes()
        size += self.matrix_factorization.approximate_size_bytes()
        return size

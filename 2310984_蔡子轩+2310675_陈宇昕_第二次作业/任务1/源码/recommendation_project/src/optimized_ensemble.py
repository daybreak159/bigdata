"""Optimized linear ensemble found during exploratory validation."""

from __future__ import annotations

import sys
from typing import List, Sequence

from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from user_cf import UserCFModel
from utils import Prediction, TestPair, TrainRecord


class OptimizedEnsembleModel:
    """
    Three-model linear ensemble.

    The weights come from validation on Train.txt only:
        0.43 * tuned ItemCF + 0.45 * tuned 48D MF + 0.12 * UserCF
    """

    def __init__(
        self,
        itemcf_weight: float = 0.43,
        mf_weight: float = 0.45,
        usercf_weight: float = 0.12,
        seed: int = 2026,
    ) -> None:
        total = itemcf_weight + mf_weight + usercf_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("ensemble weights must sum to 1.")

        self.itemcf_weight = itemcf_weight
        self.mf_weight = mf_weight
        self.usercf_weight = usercf_weight
        self.seed = seed

        self.itemcf = ItemCFModel(
            top_k=40,
            similarity_top_n=320,
            shrinkage=120.0,
            min_common=2,
            block_size=512,
            baseline_reg_user=6.0,
            baseline_reg_item=10.0,
            seed=seed,
        )
        self.matrix_factorization = MatrixFactorizationModel(
            n_factors=48,
            n_epochs=12,
            learning_rate=0.002,
            reg=0.3,
            reg_bias=0.05,
            init_std=0.01,
            init_with_baseline=True,
            freeze_bias=True,
            seed=seed,
        )
        self.usercf = UserCFModel(
            top_k=40,
            similarity_top_n=400,
            shrinkage=30.0,
            min_common=2,
            seed=seed,
        )
        self.min_rating: float = 0.0
        self.max_rating: float = 0.0

    def fit(self, train_data: Sequence[TrainRecord]) -> "OptimizedEnsembleModel":
        self.itemcf.fit(train_data)
        self.matrix_factorization.fit(train_data)
        self.usercf.fit(train_data)
        self.min_rating = self.itemcf.min_rating
        self.max_rating = self.itemcf.max_rating
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        itemcf_score = self.itemcf.predict(user_id, item_id)
        mf_score = self.matrix_factorization.predict(user_id, item_id)
        usercf_score = self.usercf.predict(user_id, item_id)
        return (
            self.itemcf_weight * itemcf_score
            + self.mf_weight * mf_score
            + self.usercf_weight * usercf_score
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
        size += self.usercf.approximate_size_bytes()
        return size

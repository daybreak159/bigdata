"""User-based collaborative filtering on baseline residuals."""

from __future__ import annotations

import sys
from typing import Dict, List, Sequence

import numpy as np

from baseline import BaselineModel
from utils import Prediction, TestPair, TrainRecord


class UserCFModel:
    """
    User-based collaborative filtering on baseline residuals.

    The model learns user-user similarities from residual vectors and predicts
    a target rating by borrowing residuals from similar users who rated the
    target item.

    Neighbor table is stored as two compact NumPy arrays per user
    (int32 IDs + float32 similarities), sorted by |similarity| descending,
    instead of nested Python dicts, to reduce memory overhead.
    """

    def __init__(
        self,
        top_k: int = 40,
        similarity_top_n: int | None = None,
        shrinkage: float = 30.0,
        min_common: int = 3,
        baseline_reg_user: float = 10.0,
        baseline_reg_item: float = 15.0,
        baseline_n_iters: int = 15,
        seed: int = 2026,
    ) -> None:
        self.top_k = top_k
        self.similarity_top_n = similarity_top_n or max(50, top_k * 4)
        self.shrinkage = shrinkage
        self.min_common = min_common
        self.baseline_reg_user = baseline_reg_user
        self.baseline_reg_item = baseline_reg_item
        self.baseline_n_iters = baseline_n_iters
        self.seed = seed

        self.baseline = BaselineModel(
            reg_user=baseline_reg_user,
            reg_item=baseline_reg_item,
            n_iters=baseline_n_iters,
            seed=seed,
        )
        # Compact neighbor storage: user_id -> sorted arrays of (neighbor_ids, similarities)
        self.user_neighbor_ids: Dict[int, np.ndarray] = {}   # dtype int32
        self.user_neighbor_sims: Dict[int, np.ndarray] = {}  # dtype float32
        self.item_ratings: Dict[int, Dict[int, float]] = {}
        self.min_rating: float = 0.0
        self.max_rating: float = 0.0

    def fit(self, train_data: Sequence[TrainRecord]) -> "UserCFModel":
        if not train_data:
            raise ValueError("train_data must not be empty")

        self.baseline.fit(train_data)
        self.min_rating = self.baseline.min_rating
        self.max_rating = self.baseline.max_rating

        user_ids = sorted({user_id for user_id, _, _ in train_data})
        item_ids = sorted({item_id for _, item_id, _ in train_data})
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

        n_users = len(user_ids)
        n_items = len(item_ids)
        residual_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        mask_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        self.item_ratings = {item_id: {} for item_id in item_ids}

        for user_id, item_id, rating in train_data:
            user_idx = user_to_idx[user_id]
            item_idx = item_to_idx[item_id]
            baseline_score = self.baseline.predict(user_id, item_id)
            residual_matrix[user_idx, item_idx] = rating - baseline_score
            mask_matrix[user_idx, item_idx] = 1.0
            self.item_ratings[item_id][user_id] = rating

        norms = np.sqrt(np.sum(residual_matrix * residual_matrix, axis=1))
        numerators = residual_matrix @ residual_matrix.T
        common_counts = mask_matrix @ mask_matrix.T
        self.user_neighbor_ids = {}
        self.user_neighbor_sims = {}
        user_ids_arr = np.array(user_ids, dtype=np.int32)

        for user_idx, user_id in enumerate(user_ids):
            denom = norms[user_idx] * norms
            valid = (denom > 1e-12) & (common_counts[user_idx] >= self.min_common)
            sims = np.zeros(n_users, dtype=np.float32)

            if np.any(valid):
                sims[valid] = numerators[user_idx, valid] / denom[valid]
                if self.shrinkage > 0:
                    sims[valid] *= (
                        common_counts[user_idx, valid]
                        / (common_counts[user_idx, valid] + self.shrinkage)
                    )
            sims[user_idx] = 0.0

            nonzero_indices = np.flatnonzero(np.abs(sims) > 1e-12)
            if nonzero_indices.size == 0:
                continue
            if nonzero_indices.size > self.similarity_top_n:
                abs_values = np.abs(sims[nonzero_indices])
                local_idx = np.argpartition(abs_values, -self.similarity_top_n)[
                    -self.similarity_top_n :
                ]
                nonzero_indices = nonzero_indices[local_idx]

            order = np.argsort(np.abs(sims[nonzero_indices]))[::-1]
            ordered_indices = nonzero_indices[order]
            self.user_neighbor_ids[user_id] = user_ids_arr[ordered_indices].copy()
            self.user_neighbor_sims[user_id] = sims[ordered_indices].copy()

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        baseline_score = self.baseline.predict(user_id, item_id)
        neighbor_ids = self.user_neighbor_ids.get(user_id)
        item_history = self.item_ratings.get(item_id)

        if neighbor_ids is None or not item_history:
            return baseline_score

        neighbor_sims = self.user_neighbor_sims[user_id]
        # Arrays are pre-sorted by |sim| desc — collect top_k matches with early exit.
        count = 0
        numerator = 0.0
        denominator = 0.0
        for i in range(len(neighbor_ids)):
            nid = int(neighbor_ids[i])
            rating = item_history.get(nid)
            if rating is None:
                continue
            sim = float(neighbor_sims[i])
            if abs(sim) <= 1e-12:
                continue
            neighbor_baseline = self.baseline.predict(nid, item_id)
            numerator += sim * (rating - neighbor_baseline)
            denominator += abs(sim)
            count += 1
            if count >= self.top_k:
                break

        if denominator <= 1e-12:
            return baseline_score
        return baseline_score + numerator / denominator

    def batch_predict(self, test_pairs: Sequence[TestPair]) -> List[Prediction]:
        return [
            (user_id, item_id, self.predict(user_id, item_id))
            for user_id, item_id in test_pairs
        ]

    def approximate_size_bytes(self) -> int:
        size = sys.getsizeof(self)
        size += self.baseline.approximate_size_bytes()
        size += sys.getsizeof(self.item_ratings)
        size += sys.getsizeof(self.user_neighbor_ids)
        size += sys.getsizeof(self.user_neighbor_sims)

        for item_id, ratings in self.item_ratings.items():
            size += sys.getsizeof(item_id) + sys.getsizeof(ratings)
            for user_id, rating in ratings.items():
                size += sys.getsizeof(user_id) + sys.getsizeof(rating)

        for user_id, arr in self.user_neighbor_ids.items():
            size += sys.getsizeof(user_id) + arr.nbytes + self.user_neighbor_sims[user_id].nbytes

        return size

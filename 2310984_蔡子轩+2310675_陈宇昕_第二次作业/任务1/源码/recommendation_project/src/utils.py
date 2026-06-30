"""Utility helpers for the course recommendation project."""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


TrainRecord = Tuple[int, int, float]
TestPair = Tuple[int, int]
Prediction = Tuple[int, int, float]


def read_train(path: str) -> List[TrainRecord]:
    """Read Train.txt and return a flat list of (user_id, item_id, rating)."""
    data: List[TrainRecord] = []
    with open(path, "r", encoding="utf-8") as file:
        while True:
            line = file.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            user_text, count_text = line.split("|")
            user_id = int(user_text)
            count = int(count_text)
            for _ in range(count):
                item_text, score_text = file.readline().split()
                data.append((user_id, int(item_text), float(score_text)))
    return data


def read_test(path: str) -> List[TestPair]:
    """Read Test.txt and return a flat list of (user_id, item_id)."""
    data: List[TestPair] = []
    with open(path, "r", encoding="utf-8") as file:
        while True:
            line = file.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            user_text, count_text = line.split("|")
            user_id = int(user_text)
            count = int(count_text)
            for _ in range(count):
                item_text = file.readline().strip()
                if item_text:
                    data.append((user_id, int(item_text)))
    return data


def read_result_form(path: str) -> List[Tuple[int, int]]:
    """
    Read a result-form file and extract real user-count blocks when possible.

    The provided ResultForm.txt in this dataset looks like a format example
    instead of a full template, so callers should validate the total count
    before relying on the returned blocks.
    """

    blocks: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    index = 0
    while index < len(lines):
        line = lines[index]
        if "|" not in line:
            index += 1
            continue
        left, right = line.split("|", 1)
        if left.isdigit() and right.isdigit():
            blocks.append((int(left), int(right)))
            index += int(right) + 1
        else:
            index += 1
    return blocks


def write_predictions(
    path: str,
    predictions: Sequence[Prediction],
    result_form_path: str | None = None,
) -> None:
    """Write predictions in the required `user|count` block format."""

    predictions = list(predictions)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    template_blocks: List[Tuple[int, int]] = []
    if result_form_path and os.path.exists(result_form_path):
        template_blocks = read_result_form(result_form_path)

    use_template = bool(template_blocks) and (
        sum(count for _, count in template_blocks) == len(predictions)
    )

    with open(path, "w", encoding="utf-8") as file:
        if use_template:
            cursor = 0
            for user_id, count in template_blocks:
                file.write(f"{user_id}|{count}\n")
                for _ in range(count):
                    _, item_id, score = predictions[cursor]
                    cursor += 1
                    file.write(f"{item_id} {score:.4f}\n")
            return

        current_user = None
        current_items: List[Tuple[int, float]] = []

        def flush_block() -> None:
            if current_user is None:
                return
            file.write(f"{current_user}|{len(current_items)}\n")
            for item_id, score in current_items:
                file.write(f"{item_id} {score:.4f}\n")

        for user_id, item_id, score in predictions:
            if current_user is None:
                current_user = user_id
            if user_id != current_user:
                flush_block()
                current_user = user_id
                current_items = []
            current_items.append((item_id, score))
        flush_block()


def train_valid_split(
    data: Sequence[TrainRecord],
    valid_ratio: float = 0.2,
    seed: int = 2026,
) -> Tuple[List[TrainRecord], List[TrainRecord]]:
    """
    Split ratings into train/valid with a user-wise strategy.

    For each user, at least one record is kept in the training set whenever
    possible so the validation stage is less likely to contain only unseen
    users.
    """

    by_user: Dict[int, List[TrainRecord]] = defaultdict(list)
    for record in data:
        by_user[record[0]].append(record)

    rng = random.Random(seed)
    train_data: List[TrainRecord] = []
    valid_data: List[TrainRecord] = []

    for user_id in sorted(by_user):
        records = by_user[user_id][:]
        rng.shuffle(records)
        if len(records) <= 1:
            train_data.extend(records)
            continue
        valid_count = max(1, int(len(records) * valid_ratio))
        valid_count = min(valid_count, len(records) - 1)
        valid_data.extend(records[:valid_count])
        train_data.extend(records[valid_count:])

    return train_data, valid_data


def build_user_k_folds(
    data: Sequence[TrainRecord],
    n_splits: int = 5,
    seed: int = 2026,
) -> List[List[TrainRecord]]:
    """
    Build user-aware K folds for cross validation.

    Ratings of the same user are shuffled and then distributed round-robin into
    different folds, which reduces the chance that a validation fold contains
    only completely unseen users.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    by_user: Dict[int, List[TrainRecord]] = defaultdict(list)
    for record in data:
        by_user[record[0]].append(record)

    rng = random.Random(seed)
    folds: List[List[TrainRecord]] = [[] for _ in range(n_splits)]

    for user_id in sorted(by_user):
        records = by_user[user_id][:]
        rng.shuffle(records)
        start_fold = rng.randrange(n_splits)
        for offset, record in enumerate(records):
            fold_index = (start_fold + offset) % n_splits
            folds[fold_index].append(record)

    return folds


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    if not y_true:
        return 0.0
    mse = sum((truth - pred) ** 2 for truth, pred in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def clip_rating(score: float, min_rating: float, max_rating: float) -> float:
    """Clamp a score into the valid rating interval."""
    return max(min_rating, min(max_rating, score))


def dataset_statistics(train_data: Sequence[TrainRecord]) -> Dict[str, float]:
    """Return simple dataset statistics for report writing."""
    if not train_data:
        return {
            "num_users": 0,
            "num_items": 0,
            "num_ratings": 0,
            "min_rating": 0.0,
            "max_rating": 0.0,
            "mean_rating": 0.0,
            "density": 0.0,
        }

    users = set()
    items = set()
    ratings = []
    for user_id, item_id, score in train_data:
        users.add(user_id)
        items.add(item_id)
        ratings.append(score)

    num_users = len(users)
    num_items = len(items)
    num_ratings = len(train_data)
    density = num_ratings / (num_users * num_items) if num_users and num_items else 0.0

    return {
        "num_users": num_users,
        "num_items": num_items,
        "num_ratings": num_ratings,
        "min_rating": min(ratings),
        "max_rating": max(ratings),
        "mean_rating": sum(ratings) / len(ratings),
        "density": density,
    }

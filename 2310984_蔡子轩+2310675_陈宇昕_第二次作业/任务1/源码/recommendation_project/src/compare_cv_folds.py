"""Compare different cross-validation fold counts for the final ensemble."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List

from ensemble import EnsembleModel
from utils import build_user_k_folds, clip_rating, read_train, rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 2026
FOLD_COUNTS = [3, 5, 6, 8, 10]


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def build_final_model() -> EnsembleModel:
    return EnsembleModel(itemcf_weight=0.55, seed=SEED)


def evaluate_fold_count(full_train, n_folds: int) -> Dict[str, float | int | str]:
    folds = build_user_k_folds(full_train, n_splits=n_folds, seed=SEED)
    fold_rmses: List[float] = []
    train_seconds_list: List[float] = []
    predict_seconds_list: List[float] = []

    for valid_index in range(n_folds):
        valid_data = folds[valid_index]
        train_data = [
            record
            for fold_index, fold in enumerate(folds)
            if fold_index != valid_index
            for record in fold
        ]
        valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
        y_true = [rating for _, _, rating in valid_data]

        model = build_final_model()

        train_start = time.perf_counter()
        model.fit(train_data)
        train_seconds = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        predictions = model.batch_predict(valid_pairs)
        predict_seconds = time.perf_counter() - predict_start

        y_pred = [
            clip_rating(score, model.min_rating, model.max_rating)
            for _, _, score in predictions
        ]
        fold_rmses.append(rmse(y_true, y_pred))
        train_seconds_list.append(train_seconds)
        predict_seconds_list.append(predict_seconds)

        print(
            f"{n_folds}-fold {valid_index + 1}/{n_folds}: "
            f"rmse={fold_rmses[-1]:.6f} "
            f"train={train_seconds:.4f}s predict={predict_seconds:.4f}s",
            flush=True,
        )

    row: Dict[str, float | int | str] = {
        "n_folds": n_folds,
        "fold_sizes": ";".join(str(len(fold)) for fold in folds),
        "mean_rmse": mean(fold_rmses),
        "std_rmse": std(fold_rmses),
        "total_train_seconds": sum(train_seconds_list),
        "avg_train_seconds": mean(train_seconds_list),
        "total_predict_seconds": sum(predict_seconds_list),
        "avg_predict_seconds": mean(predict_seconds_list),
    }
    for fold_index, fold_rmse in enumerate(fold_rmses, start=1):
        row[f"fold_{fold_index}_rmse"] = fold_rmse
    return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_train = read_train(str(DATA_DIR / "Train.txt"))
    rows = []

    for n_folds in FOLD_COUNTS:
        print(f"\nRunning {n_folds}-fold cross validation", flush=True)
        rows.append(evaluate_fold_count(full_train, n_folds))

    output_path = RESULTS_DIR / "cv_fold_count_results.csv"
    fieldnames = [
        "n_folds",
        "fold_sizes",
        "mean_rmse",
        "std_rmse",
        "total_train_seconds",
        "avg_train_seconds",
        "total_predict_seconds",
        "avg_predict_seconds",
        *[f"fold_{fold_index}_rmse" for fold_index in range(1, max(FOLD_COUNTS) + 1)],
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{row[key]:.6f}"
                        if isinstance(row.get(key), float)
                        else row.get(key, "")
                    )
                    for key in fieldnames
                }
            )

    print()
    print(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    main()

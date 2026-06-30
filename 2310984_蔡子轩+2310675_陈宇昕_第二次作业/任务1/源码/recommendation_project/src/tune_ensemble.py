"""Fine-grained cross-validation search for ensemble weights."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List

from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from utils import build_user_k_folds, clip_rating, read_train, rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 2026
N_FOLDS = 5


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def build_itemcf() -> ItemCFModel:
    return ItemCFModel(
        top_k=40,
        similarity_top_n=200,
        shrinkage=50.0,
        min_common=2,
        block_size=256,
        seed=SEED,
    )


def build_mf() -> MatrixFactorizationModel:
    return MatrixFactorizationModel(
        n_factors=4,
        n_epochs=12,
        learning_rate=0.002,
        reg=0.2,
        reg_bias=0.05,
        init_std=0.01,
        init_with_baseline=True,
        freeze_bias=True,
        seed=SEED,
    )


def weight_grid() -> List[float]:
    coarse = [round(value / 10, 2) for value in range(0, 11)]
    fine = [round(value / 100, 2) for value in range(45, 66)]
    return sorted(set(coarse + fine))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_train = read_train(str(DATA_DIR / "Train.txt"))
    folds = build_user_k_folds(full_train, n_splits=N_FOLDS, seed=SEED)
    weights = weight_grid()

    rows_by_weight: Dict[float, Dict[str, float]] = {
        weight: {"itemcf_weight": weight} for weight in weights
    }
    train_seconds_list: List[float] = []
    predict_seconds_list: List[float] = []
    memory_bytes_list: List[float] = []

    for valid_index in range(N_FOLDS):
        valid_data = folds[valid_index]
        train_data = [
            record
            for fold_index, fold in enumerate(folds)
            if fold_index != valid_index
            for record in fold
        ]
        valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
        y_true = [rating for _, _, rating in valid_data]

        itemcf = build_itemcf()
        mf = build_mf()

        train_start = time.perf_counter()
        itemcf.fit(train_data)
        mf.fit(train_data)
        train_seconds = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        itemcf_predictions = itemcf.batch_predict(valid_pairs)
        mf_predictions = mf.batch_predict(valid_pairs)
        predict_seconds = time.perf_counter() - predict_start

        itemcf_scores = [score for _, _, score in itemcf_predictions]
        mf_scores = [score for _, _, score in mf_predictions]
        rating_min = itemcf.min_rating
        rating_max = itemcf.max_rating

        for weight in weights:
            y_pred = [
                clip_rating(
                    weight * itemcf_score + (1.0 - weight) * mf_score,
                    rating_min,
                    rating_max,
                )
                for itemcf_score, mf_score in zip(itemcf_scores, mf_scores)
            ]
            rows_by_weight[weight][f"fold_{valid_index + 1}_rmse"] = rmse(y_true, y_pred)

        train_seconds_list.append(train_seconds)
        predict_seconds_list.append(predict_seconds)
        memory_bytes_list.append(
            float(itemcf.approximate_size_bytes() + mf.approximate_size_bytes())
        )
        print(
            f"Fold {valid_index + 1}/{N_FOLDS}: "
            f"train={train_seconds:.4f}s predict={predict_seconds:.4f}s"
        )

    rows: List[Dict[str, float]] = []
    for weight in weights:
        row = rows_by_weight[weight]
        fold_rmses = [row[f"fold_{fold_index}_rmse"] for fold_index in range(1, N_FOLDS + 1)]
        row["mean_rmse"] = mean(fold_rmses)
        row["std_rmse"] = std(fold_rmses)
        row["avg_train_seconds"] = mean(train_seconds_list)
        row["avg_predict_seconds"] = mean(predict_seconds_list)
        row["avg_memory_bytes"] = mean(memory_bytes_list)
        row["avg_memory_mb"] = mean(memory_bytes_list) / (1024 * 1024)
        rows.append(row)

    output_path = RESULTS_DIR / "ensemble_weight_fine_results.csv"
    fieldnames = [
        "itemcf_weight",
        *[f"fold_{fold_index}_rmse" for fold_index in range(1, N_FOLDS + 1)],
        "mean_rmse",
        "std_rmse",
        "avg_train_seconds",
        "avg_predict_seconds",
        "avg_memory_bytes",
        "avg_memory_mb",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{row[key]:.6f}"
                        if key.startswith("fold_")
                        or key
                        in {
                            "mean_rmse",
                            "std_rmse",
                            "avg_train_seconds",
                            "avg_predict_seconds",
                            "avg_memory_mb",
                        }
                        else int(row[key])
                        if key == "avg_memory_bytes"
                        else row[key]
                    )
                    for key in fieldnames
                }
            )

    best = min(rows, key=lambda row: row["mean_rmse"])
    print()
    print(
        "Best ensemble weight: "
        f"{best['itemcf_weight']:.2f} "
        f"(mean RMSE={best['mean_rmse']:.6f}, std={best['std_rmse']:.6f})"
    )
    print(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    main()

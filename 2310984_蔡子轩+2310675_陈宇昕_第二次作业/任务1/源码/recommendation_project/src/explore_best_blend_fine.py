"""Fine search around the best exploratory three-model blend."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Callable, Sequence

from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from user_cf import UserCFModel
from utils import TrainRecord, build_user_k_folds, clip_rating, read_train, rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 2026
N_FOLDS = 5


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def configs() -> list[tuple[str, Callable[[], object]]]:
    return [
        (
            "ItemCFb",
            lambda: ItemCFModel(
                top_k=40,
                similarity_top_n=320,
                shrinkage=120.0,
                min_common=2,
                block_size=512,
                baseline_reg_user=6.0,
                baseline_reg_item=10.0,
                seed=SEED,
            ),
        ),
        (
            "MF",
            lambda: MatrixFactorizationModel(
                n_factors=48,
                n_epochs=12,
                learning_rate=0.002,
                reg=0.3,
                reg_bias=0.05,
                init_std=0.01,
                init_with_baseline=True,
                freeze_bias=True,
                seed=SEED,
            ),
        ),
        (
            "UserCF",
            lambda: UserCFModel(
                top_k=40,
                similarity_top_n=400,
                shrinkage=30.0,
                min_common=2,
                seed=SEED,
            ),
        ),
    ]


def fit_predict(
    build_model: Callable[[], object],
    train_data: Sequence[TrainRecord],
    valid_data: Sequence[TrainRecord],
) -> tuple[list[float], float, float]:
    valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
    model = build_model()
    start = time.perf_counter()
    model.fit(train_data)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = model.batch_predict(valid_pairs)
    predict_seconds = time.perf_counter() - start
    print(
        f"trained {type(model).__name__}: "
        f"train={train_seconds:.4f}s predict={predict_seconds:.4f}s",
        flush=True,
    )
    return [score for _, _, score in predictions], model.min_rating, model.max_rating


def weight_grid() -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for itemcf_percent in range(40, 61):
        for mf_percent in range(25, 46):
            user_percent = 100 - itemcf_percent - mf_percent
            if 5 <= user_percent <= 25:
                rows.append(
                    (
                        itemcf_percent / 100,
                        mf_percent / 100,
                        user_percent / 100,
                    )
                )
    return rows


def blended_rmse(
    y_true: list[float],
    scores_by_model: list[list[float]],
    weights: tuple[float, float, float],
    rating_min: float,
    rating_max: float,
) -> float:
    y_pred = []
    for scores in zip(*scores_by_model):
        score = sum(weight * pred for weight, pred in zip(weights, scores))
        y_pred.append(clip_rating(score, rating_min, rating_max))
    return rmse(y_true, y_pred)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "exploration_best_blend_fine.csv"
    full_train = read_train(str(DATA_DIR / "Train.txt"))
    folds = build_user_k_folds(full_train, n_splits=N_FOLDS, seed=SEED)

    fold_truths: list[list[float]] = []
    fold_scores: list[list[list[float]]] = []
    fold_ranges: list[tuple[float, float]] = []

    for valid_index in range(N_FOLDS):
        valid_data = folds[valid_index]
        train_data = [
            record
            for fold_index, fold in enumerate(folds)
            if fold_index != valid_index
            for record in fold
        ]
        y_true = [rating for _, _, rating in valid_data]
        fold_truths.append(y_true)
        current_scores = []
        rating_min = float("inf")
        rating_max = float("-inf")
        print(f"fold {valid_index + 1}/{N_FOLDS}", flush=True)
        for _, builder in configs():
            scores, model_min, model_max = fit_predict(builder, train_data, valid_data)
            current_scores.append(scores)
            rating_min = min(rating_min, model_min)
            rating_max = max(rating_max, model_max)
        fold_scores.append(current_scores)
        fold_ranges.append((rating_min, rating_max))

    rows = []
    for weights in weight_grid():
        fold_rmses = []
        for fold_index in range(N_FOLDS):
            rating_min, rating_max = fold_ranges[fold_index]
            fold_rmses.append(
                blended_rmse(
                    fold_truths[fold_index],
                    fold_scores[fold_index],
                    weights,
                    rating_min,
                    rating_max,
                )
            )
        rows.append(
            {
                "itemcf_weight": weights[0],
                "mf_weight": weights[1],
                "usercf_weight": weights[2],
                "mean_rmse": mean(fold_rmses),
                "std_rmse": std(fold_rmses),
                **{
                    f"fold_{fold_index}_rmse": fold_rmse
                    for fold_index, fold_rmse in enumerate(fold_rmses, start=1)
                },
            }
        )

    rows.sort(key=lambda row: row["mean_rmse"])
    fieldnames = [
        "itemcf_weight",
        "mf_weight",
        "usercf_weight",
        *[f"fold_{fold_index}_rmse" for fold_index in range(1, N_FOLDS + 1)],
        "mean_rmse",
        "std_rmse",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.6f}" for key in fieldnames})

    print("Top 20 fine weights:")
    for row in rows[:20]:
        print(
            f"{row['mean_rmse']:.6f} "
            f"{row['itemcf_weight']:.2f}+{row['mf_weight']:.2f}+"
            f"{row['usercf_weight']:.2f}",
            flush=True,
        )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

"""Explore validation blends among tuned recommendation models."""

from __future__ import annotations

import csv
import math
import time
from itertools import product
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


def model_configs() -> list[tuple[str, Callable[[], object]]]:
    return [
        (
            "ItemCF_tuned",
            lambda: ItemCFModel(
                top_k=40,
                similarity_top_n=320,
                shrinkage=120.0,
                min_common=2,
                block_size=512,
                seed=SEED,
            ),
        ),
        (
            "ItemCF_tuned_baseline_ru6_ri10",
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
            "MF_tuned",
            lambda: MatrixFactorizationModel(
                n_factors=8,
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
            "MF_tuned_baseline_ru6_ri10",
            lambda: MatrixFactorizationModel(
                n_factors=8,
                n_epochs=12,
                learning_rate=0.002,
                reg=0.3,
                reg_bias=0.05,
                init_std=0.01,
                init_with_baseline=True,
                freeze_bias=True,
                baseline_reg_user=6.0,
                baseline_reg_item=10.0,
                seed=SEED,
            ),
        ),
        (
            "UserCF_tuned",
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
) -> tuple[list[float], float, float, float, float, float]:
    valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
    model = build_model()

    start = time.perf_counter()
    model.fit(train_data)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.batch_predict(valid_pairs)
    predict_seconds = time.perf_counter() - start

    scores = [score for _, _, score in predictions]
    memory_mb = model.approximate_size_bytes() / (1024 * 1024)
    return (
        scores,
        train_seconds,
        predict_seconds,
        memory_mb,
        model.min_rating,
        model.max_rating,
    )


def pair_weights() -> list[tuple[float, float]]:
    weights = [round(value / 100, 2) for value in range(0, 101)]
    return [(weight, round(1.0 - weight, 2)) for weight in weights]


def triple_weights(step: int = 5) -> list[tuple[float, float, float]]:
    weights: list[tuple[float, float, float]] = []
    for first in range(0, 101, step):
        for second in range(0, 101 - first, step):
            third = 100 - first - second
            weights.append((first / 100, second / 100, third / 100))
    return weights


def score_blend(
    y_true: list[float],
    model_scores: list[list[float]],
    weights: tuple[float, ...],
    rating_min: float,
    rating_max: float,
) -> float:
    y_pred = []
    for scores in zip(*model_scores):
        score = sum(weight * pred for weight, pred in zip(weights, scores))
        y_pred.append(clip_rating(score, rating_min, rating_max))
    return rmse(y_true, y_pred)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "exploration_blend_selected.csv"
    full_train = read_train(str(DATA_DIR / "Train.txt"))
    folds = build_user_k_folds(full_train, n_splits=N_FOLDS, seed=SEED)
    configs = model_configs()

    fold_truths: list[list[float]] = []
    fold_scores: list[dict[str, list[float]]] = []
    fold_rating_ranges: list[tuple[float, float]] = []
    timings: dict[str, dict[str, list[float]]] = {
        name: {"train": [], "predict": [], "memory": []}
        for name, _ in configs
    }

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
        fold_scores.append({})

        range_min = float("inf")
        range_max = float("-inf")
        for name, builder in configs:
            scores, train_sec, pred_sec, memory_mb, rating_min, rating_max = fit_predict(
                builder,
                train_data,
                valid_data,
            )
            fold_scores[-1][name] = scores
            timings[name]["train"].append(train_sec)
            timings[name]["predict"].append(pred_sec)
            timings[name]["memory"].append(memory_mb)
            range_min = min(range_min, rating_min)
            range_max = max(range_max, rating_max)
            single_rmse = score_blend(y_true, [scores], (1.0,), rating_min, rating_max)
            print(
                f"fold {valid_index + 1}/{N_FOLDS} {name}: "
                f"rmse={single_rmse:.6f}",
                flush=True,
            )
        fold_rating_ranges.append((range_min, range_max))

    rows: list[dict[str, object]] = []

    pair_sets = [
        ("Pair_ItemCF_MF", ("ItemCF_tuned", "MF_tuned")),
        (
            "Pair_ItemCFb_MFb",
            ("ItemCF_tuned_baseline_ru6_ri10", "MF_tuned_baseline_ru6_ri10"),
        ),
        ("Pair_ItemCF_MFb", ("ItemCF_tuned", "MF_tuned_baseline_ru6_ri10")),
        ("Pair_ItemCFb_MF", ("ItemCF_tuned_baseline_ru6_ri10", "MF_tuned")),
    ]
    for blend_name, names in pair_sets:
        for weights in pair_weights():
            fold_rmses = []
            for fold_index in range(N_FOLDS):
                rating_min, rating_max = fold_rating_ranges[fold_index]
                fold_rmses.append(
                    score_blend(
                        fold_truths[fold_index],
                        [fold_scores[fold_index][name] for name in names],
                        weights,
                        rating_min,
                        rating_max,
                    )
                )
            rows.append(
                {
                    "blend": blend_name,
                    "models": "+".join(names),
                    "weights": "+".join(f"{weight:.2f}" for weight in weights),
                    "mean_rmse": mean(fold_rmses),
                    "std_rmse": std(fold_rmses),
                    **{
                        f"fold_{fold_index}_rmse": fold_rmse
                        for fold_index, fold_rmse in enumerate(fold_rmses, start=1)
                    },
                }
            )

    triple_sets = [
        ("Triple_ItemCF_MF_UserCF", ("ItemCF_tuned", "MF_tuned", "UserCF_tuned")),
        (
            "Triple_ItemCFb_MF_UserCF",
            ("ItemCF_tuned_baseline_ru6_ri10", "MF_tuned", "UserCF_tuned"),
        ),
        (
            "Triple_ItemCF_MFb_UserCF",
            ("ItemCF_tuned", "MF_tuned_baseline_ru6_ri10", "UserCF_tuned"),
        ),
        (
            "Triple_ItemCFb_MFb_UserCF",
            (
                "ItemCF_tuned_baseline_ru6_ri10",
                "MF_tuned_baseline_ru6_ri10",
                "UserCF_tuned",
            ),
        ),
    ]
    for blend_name, triple_names in triple_sets:
        for weights in triple_weights(step=5):
            fold_rmses = []
            for fold_index in range(N_FOLDS):
                rating_min, rating_max = fold_rating_ranges[fold_index]
                fold_rmses.append(
                    score_blend(
                        fold_truths[fold_index],
                        [fold_scores[fold_index][name] for name in triple_names],
                        weights,
                        rating_min,
                        rating_max,
                    )
                )
            rows.append(
                {
                    "blend": blend_name,
                    "models": "+".join(triple_names),
                    "weights": "+".join(f"{weight:.2f}" for weight in weights),
                    "mean_rmse": mean(fold_rmses),
                    "std_rmse": std(fold_rmses),
                    **{
                        f"fold_{fold_index}_rmse": fold_rmse
                        for fold_index, fold_rmse in enumerate(fold_rmses, start=1)
                    },
                }
            )

    rows.sort(key=lambda row: float(row["mean_rmse"]))
    fieldnames = [
        "blend",
        "models",
        "weights",
        *[f"fold_{fold_index}_rmse" for fold_index in range(1, N_FOLDS + 1)],
        "mean_rmse",
        "std_rmse",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{row[key]:.6f}"
                    if key.startswith("fold_") or key in {"mean_rmse", "std_rmse"}
                    else row[key]
                    for key in fieldnames
                }
            )

    print()
    print("Top 20 blends:")
    for row in rows[:20]:
        print(
            f"{row['mean_rmse']:.6f} {row['blend']} "
            f"{row['weights']} {row['models']}",
            flush=True,
        )
    print(f"Saved to {output_path}")

    print()
    print("Average model costs:")
    for name in timings:
        print(
            f"{name}: train={mean(timings[name]['train']):.4f}s "
            f"predict={mean(timings[name]['predict']):.4f}s "
            f"memory={mean(timings[name]['memory']):.4f}MB",
            flush=True,
        )


if __name__ == "__main__":
    main()

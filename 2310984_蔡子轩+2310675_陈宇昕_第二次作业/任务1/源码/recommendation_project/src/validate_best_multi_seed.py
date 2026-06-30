"""Validate best exploratory methods across multiple random CV seeds."""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Callable, Sequence

from baseline import BaselineModel
from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from user_cf import UserCFModel
from utils import TrainRecord, build_user_k_folds, clip_rating, read_train, rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
N_FOLDS = 5
SEEDS = [2024, 2025, 2026, 2027, 2028]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def model_configs(seed: int) -> list[tuple[str, Callable[[], object]]]:
    return [
        (
            "Baseline_default",
            lambda: BaselineModel(seed=seed),
        ),
        (
            "Baseline_tuned_ru6_ri10",
            lambda: BaselineModel(reg_user=6.0, reg_item=10.0, n_iters=15, seed=seed),
        ),
        (
            "ItemCF_official",
            lambda: ItemCFModel(
                top_k=40,
                similarity_top_n=200,
                shrinkage=50.0,
                min_common=2,
                block_size=512,
                seed=seed,
            ),
        ),
        (
            "ItemCF_tuned_baseline",
            lambda: ItemCFModel(
                top_k=40,
                similarity_top_n=320,
                shrinkage=120.0,
                min_common=2,
                block_size=512,
                baseline_reg_user=6.0,
                baseline_reg_item=10.0,
                seed=seed,
            ),
        ),
        (
            "MF_official",
            lambda: MatrixFactorizationModel(
                n_factors=4,
                n_epochs=12,
                learning_rate=0.002,
                reg=0.2,
                reg_bias=0.05,
                init_std=0.01,
                init_with_baseline=True,
                freeze_bias=True,
                seed=seed,
            ),
        ),
        (
            "MF_tuned",
            lambda: MatrixFactorizationModel(
                n_factors=48,
                n_epochs=12,
                learning_rate=0.002,
                reg=0.3,
                reg_bias=0.05,
                init_std=0.01,
                init_with_baseline=True,
                freeze_bias=True,
                seed=seed,
            ),
        ),
        (
            "UserCF_tuned",
            lambda: UserCFModel(
                top_k=40,
                similarity_top_n=400,
                shrinkage=30.0,
                min_common=2,
                seed=seed,
            ),
        ),
    ]


def fit_predict(
    build_model: Callable[[], object],
    train_data: Sequence[TrainRecord],
    valid_data: Sequence[TrainRecord],
) -> dict[str, object]:
    valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
    y_true = [rating for _, _, rating in valid_data]
    model = build_model()

    start = time.perf_counter()
    model.fit(train_data)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.batch_predict(valid_pairs)
    predict_seconds = time.perf_counter() - start

    scores = [score for _, _, score in predictions]
    clipped_scores = [
        clip_rating(score, model.min_rating, model.max_rating)
        for score in scores
    ]
    return {
        "scores": scores,
        "rmse": rmse(y_true, clipped_scores),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "memory_mb": model.approximate_size_bytes() / (1024 * 1024),
        "min_rating": model.min_rating,
        "max_rating": model.max_rating,
    }


def blend_rmse(
    y_true: list[float],
    scores_by_model: list[list[float]],
    weights: tuple[float, ...],
    rating_min: float,
    rating_max: float,
) -> float:
    y_pred = []
    for scores in zip(*scores_by_model):
        score = sum(weight * pred for weight, pred in zip(weights, scores))
        y_pred.append(clip_rating(score, rating_min, rating_max))
    return rmse(y_true, y_pred)


def paired_improvement(
    rows: list[dict[str, object]],
    baseline_name: str,
    candidate_name: str,
) -> tuple[float, float, int]:
    by_key: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["fold"]))
        by_key.setdefault(key, {})[str(row["model"])] = float(row["rmse"])

    diffs = []
    for values in by_key.values():
        if baseline_name in values and candidate_name in values:
            diffs.append(values[baseline_name] - values[candidate_name])
    return mean(diffs), std(diffs), sum(1 for value in diffs if value > 0)


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    fieldnames = [
        "model",
        "mean_rmse",
        "std_rmse",
        "min_rmse",
        "max_rmse",
        "mean_train_seconds",
        "mean_predict_seconds",
        "mean_memory_mb",
        "paired_improvement_vs_official_ensemble",
        "paired_improvement_std",
        "wins_vs_official_ensemble",
        "num_evaluations",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in sorted(
            by_model,
            key=lambda name: mean([float(row["rmse"]) for row in by_model[name]]),
        ):
            model_rows = by_model[model_name]
            rmses = [float(row["rmse"]) for row in model_rows]
            trains = [float(row["train_seconds"]) for row in model_rows]
            predicts = [float(row["predict_seconds"]) for row in model_rows]
            memories = [float(row["memory_mb"]) for row in model_rows]
            improvement, improvement_std, wins = paired_improvement(
                rows,
                "Ensemble_official",
                model_name,
            )
            writer.writerow(
                {
                    "model": model_name,
                    "mean_rmse": f"{mean(rmses):.6f}",
                    "std_rmse": f"{std(rmses):.6f}",
                    "min_rmse": f"{min(rmses):.6f}",
                    "max_rmse": f"{max(rmses):.6f}",
                    "mean_train_seconds": f"{mean(trains):.4f}",
                    "mean_predict_seconds": f"{mean(predicts):.4f}",
                    "mean_memory_mb": f"{mean(memories):.4f}",
                    "paired_improvement_vs_official_ensemble": f"{improvement:.6f}",
                    "paired_improvement_std": f"{improvement_std:.6f}",
                    "wins_vs_official_ensemble": wins,
                    "num_evaluations": len(model_rows),
                }
            )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    details_path = RESULTS_DIR / "exploration_best_methods_multiseed_folds.csv"
    summary_path = RESULTS_DIR / "exploration_best_methods_multiseed_summary.csv"
    full_train = read_train(str(DATA_DIR / "Train.txt"))

    detail_fieldnames = [
        "seed",
        "fold",
        "model",
        "rmse",
        "train_seconds",
        "predict_seconds",
        "memory_mb",
    ]
    all_rows: list[dict[str, object]] = []
    completed_folds: set[tuple[int, int]] = set()

    if details_path.exists():
        with details_path.open("r", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            existing_rows = list(reader)
        counts: dict[tuple[int, int], int] = {}
        for row in existing_rows:
            all_rows.append(row)
            key = (int(row["seed"]), int(row["fold"]))
            counts[key] = counts.get(key, 0) + 1
        completed_folds = {
            key
            for key, count in counts.items()
            if count >= 10
        }

    file_exists = details_path.exists() and os.path.getsize(details_path) > 0
    with details_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detail_fieldnames)
        if not file_exists:
            writer.writeheader()

        for seed in SEEDS:
            folds = build_user_k_folds(full_train, n_splits=N_FOLDS, seed=seed)
            for valid_index in range(N_FOLDS):
                fold_number = valid_index + 1
                if (seed, fold_number) in completed_folds:
                    print(f"seed={seed} fold={fold_number}/{N_FOLDS} skipped", flush=True)
                    continue

                valid_data = folds[valid_index]
                train_data = [
                    record
                    for fold_index, fold in enumerate(folds)
                    if fold_index != valid_index
                    for record in fold
                ]
                y_true = [rating for _, _, rating in valid_data]
                model_outputs: dict[str, dict[str, object]] = {}

                print(f"seed={seed} fold={fold_number}/{N_FOLDS}", flush=True)
                for model_name, builder in model_configs(seed):
                    output = fit_predict(builder, train_data, valid_data)
                    model_outputs[model_name] = output
                    row = {
                        "seed": seed,
                        "fold": fold_number,
                        "model": model_name,
                        "rmse": f"{float(output['rmse']):.6f}",
                        "train_seconds": f"{float(output['train_seconds']):.4f}",
                        "predict_seconds": f"{float(output['predict_seconds']):.4f}",
                        "memory_mb": f"{float(output['memory_mb']):.4f}",
                    }
                    writer.writerow(row)
                    all_rows.append(row)
                    print(f"  {model_name}: rmse={float(output['rmse']):.6f}", flush=True)

                ensembles = [
                    (
                        "Ensemble_official",
                        ("ItemCF_official", "MF_official"),
                        (0.55, 0.45),
                    ),
                    (
                        "Ensemble_pair_tuned",
                        ("ItemCF_tuned_baseline", "MF_tuned"),
                        (0.59, 0.41),
                    ),
                    (
                        "Ensemble_triple_tuned",
                        ("ItemCF_tuned_baseline", "MF_tuned", "UserCF_tuned"),
                        (0.43, 0.45, 0.12),
                    ),
                ]
                for ensemble_name, component_names, weights in ensembles:
                    components = [model_outputs[name] for name in component_names]
                    rating_min = min(float(component["min_rating"]) for component in components)
                    rating_max = max(float(component["max_rating"]) for component in components)
                    ensemble_rmse = blend_rmse(
                        y_true,
                        [component["scores"] for component in components],
                        weights,
                        rating_min,
                        rating_max,
                    )
                    row = {
                        "seed": seed,
                        "fold": fold_number,
                        "model": ensemble_name,
                        "rmse": f"{ensemble_rmse:.6f}",
                        "train_seconds": f"{sum(float(component['train_seconds']) for component in components):.4f}",
                        "predict_seconds": f"{sum(float(component['predict_seconds']) for component in components):.4f}",
                        "memory_mb": f"{sum(float(component['memory_mb']) for component in components):.4f}",
                    }
                    writer.writerow(row)
                    all_rows.append(row)
                    print(f"  {ensemble_name}: rmse={ensemble_rmse:.6f}", flush=True)
                file.flush()

    write_summary(all_rows, summary_path)
    print(f"Saved fold details to {details_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

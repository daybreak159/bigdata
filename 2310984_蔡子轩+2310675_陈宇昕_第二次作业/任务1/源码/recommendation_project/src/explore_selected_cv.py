"""Run 5-fold CV for selected exploratory recommendation candidates."""

from __future__ import annotations

import csv
import math
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
SEED = 2026
N_FOLDS = 5


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def candidate_configs() -> list[tuple[str, Callable[[], object]]]:
    return [
        (
            "Baseline_default",
            lambda: BaselineModel(seed=SEED),
        ),
        (
            "Baseline_ru6_ri10",
            lambda: BaselineModel(reg_user=6.0, reg_item=10.0, n_iters=15, seed=SEED),
        ),
        (
            "ItemCF_official_k40_n200_sh50_mc2",
            lambda: ItemCFModel(
                top_k=40,
                similarity_top_n=200,
                shrinkage=50.0,
                min_common=2,
                block_size=512,
                seed=SEED,
            ),
        ),
        (
            "ItemCF_k40_n320_sh50_mc2",
            lambda: ItemCFModel(
                top_k=40,
                similarity_top_n=320,
                shrinkage=50.0,
                min_common=2,
                block_size=512,
                seed=SEED,
            ),
        ),
        (
            "ItemCF_k40_n320_sh80_mc2",
            lambda: ItemCFModel(
                top_k=40,
                similarity_top_n=320,
                shrinkage=80.0,
                min_common=2,
                block_size=512,
                seed=SEED,
            ),
        ),
        (
            "ItemCF_k40_n320_sh120_mc2",
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
            "ItemCF_k60_n320_sh120_mc2",
            lambda: ItemCFModel(
                top_k=60,
                similarity_top_n=320,
                shrinkage=120.0,
                min_common=2,
                block_size=512,
                seed=SEED,
            ),
        ),
        (
            "ItemCF_k40_n240_sh120_mc2",
            lambda: ItemCFModel(
                top_k=40,
                similarity_top_n=240,
                shrinkage=120.0,
                min_common=2,
                block_size=512,
                seed=SEED,
            ),
        ),
        (
            "UserCF_k40_n400_sh30_mc2",
            lambda: UserCFModel(
                top_k=40,
                similarity_top_n=400,
                shrinkage=30.0,
                min_common=2,
                seed=SEED,
            ),
        ),
        (
            "MF_official_f4_lr002_reg02",
            lambda: MatrixFactorizationModel(
                n_factors=4,
                n_epochs=12,
                learning_rate=0.002,
                reg=0.2,
                reg_bias=0.05,
                init_std=0.01,
                init_with_baseline=True,
                freeze_bias=True,
                seed=SEED,
            ),
        ),
        (
            "MF_f5_lr002_reg03",
            lambda: MatrixFactorizationModel(
                n_factors=5,
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
            "MF_f8_lr002_reg03",
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
    ]


def evaluate_fold(
    build_model: Callable[[], object],
    train_data: Sequence[TrainRecord],
    valid_data: Sequence[TrainRecord],
) -> tuple[float, float, float, float]:
    valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
    y_true = [rating for _, _, rating in valid_data]
    model = build_model()

    start = time.perf_counter()
    model.fit(train_data)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.batch_predict(valid_pairs)
    predict_seconds = time.perf_counter() - start

    y_pred = [
        clip_rating(score, model.min_rating, model.max_rating)
        for _, _, score in predictions
    ]
    memory_mb = model.approximate_size_bytes() / (1024 * 1024)
    return rmse(y_true, y_pred), train_seconds, predict_seconds, memory_mb


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "exploration_cv_selected.csv"
    full_train = read_train(str(DATA_DIR / "Train.txt"))
    folds = build_user_k_folds(full_train, n_splits=N_FOLDS, seed=SEED)

    fieldnames = [
        "name",
        *[f"fold_{fold_index}_rmse" for fold_index in range(1, N_FOLDS + 1)],
        "mean_rmse",
        "std_rmse",
        "avg_train_seconds",
        "avg_predict_seconds",
        "avg_memory_mb",
    ]

    best_name = ""
    best_rmse = math.inf
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for name, builder in candidate_configs():
            fold_rmses: list[float] = []
            train_seconds: list[float] = []
            predict_seconds: list[float] = []
            memory_mb: list[float] = []

            for valid_index in range(N_FOLDS):
                valid_data = folds[valid_index]
                train_data = [
                    record
                    for fold_index, fold in enumerate(folds)
                    if fold_index != valid_index
                    for record in fold
                ]
                fold_rmse, fold_train, fold_predict, fold_memory = evaluate_fold(
                    builder,
                    train_data,
                    valid_data,
                )
                fold_rmses.append(fold_rmse)
                train_seconds.append(fold_train)
                predict_seconds.append(fold_predict)
                memory_mb.append(fold_memory)
                print(
                    f"{name} fold {valid_index + 1}/{N_FOLDS}: "
                    f"rmse={fold_rmse:.6f}",
                    flush=True,
                )

            row = {
                "name": name,
                "mean_rmse": f"{mean(fold_rmses):.6f}",
                "std_rmse": f"{std(fold_rmses):.6f}",
                "avg_train_seconds": f"{mean(train_seconds):.4f}",
                "avg_predict_seconds": f"{mean(predict_seconds):.4f}",
                "avg_memory_mb": f"{mean(memory_mb):.4f}",
            }
            for fold_index, fold_rmse in enumerate(fold_rmses, start=1):
                row[f"fold_{fold_index}_rmse"] = f"{fold_rmse:.6f}"
            writer.writerow(row)
            file.flush()

            current_rmse = mean(fold_rmses)
            if current_rmse < best_rmse:
                best_name = name
                best_rmse = current_rmse
            print(
                f"candidate {name}: mean={current_rmse:.6f}; "
                f"best={best_name}:{best_rmse:.6f}",
                flush=True,
            )

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

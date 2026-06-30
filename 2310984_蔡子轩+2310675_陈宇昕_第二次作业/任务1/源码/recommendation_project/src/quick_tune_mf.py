"""Quick validation search for the biased matrix factorization model."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from matrix_factorization import MatrixFactorizationModel
from utils import TrainRecord, build_user_k_folds, clip_rating, read_train, rmse, train_valid_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 2026


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def evaluate(model: MatrixFactorizationModel, train_data: Sequence[TrainRecord], valid_data: Sequence[TrainRecord]) -> Dict[str, float]:
    valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
    y_true = [rating for _, _, rating in valid_data]

    start = time.perf_counter()
    with np.errstate(over="raise", invalid="raise"):
        model.fit(train_data)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.batch_predict(valid_pairs)
    predict_seconds = time.perf_counter() - start

    y_pred = [
        clip_rating(score, model.min_rating, model.max_rating)
        for _, _, score in predictions
    ]
    if any(not math.isfinite(score) for score in y_pred):
        raise FloatingPointError("non-finite prediction")

    return {
        "rmse": rmse(y_true, y_pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "memory_mb": model.approximate_size_bytes() / (1024 * 1024),
    }


def make_model(config: Dict[str, float | int | bool]) -> MatrixFactorizationModel:
    return MatrixFactorizationModel(
        n_factors=int(config["n_factors"]),
        n_epochs=int(config["n_epochs"]),
        learning_rate=float(config["learning_rate"]),
        reg=float(config["reg"]),
        reg_bias=float(config["reg_bias"]),
        init_std=float(config["init_std"]),
        init_with_baseline=bool(config["init_with_baseline"]),
        clip_during_training=bool(config["clip_during_training"]),
        seed=SEED,
    )


def config_rows() -> List[Dict[str, float | int | bool]]:
    rows: List[Dict[str, float | int | bool]] = []

    # Bias-initialized latent factor models, following the lecture's
    # baseline-predictor + user-item-interaction decomposition.
    for n_factors in [2, 5, 10, 20]:
        for learning_rate in [0.0005, 0.001, 0.002, 0.005]:
            for reg in [0.02, 0.05, 0.1]:
                rows.append(
                    dict(
                        n_factors=n_factors,
                        n_epochs=30,
                        learning_rate=learning_rate,
                        reg=reg,
                        reg_bias=0.05,
                        init_std=0.01,
                        init_with_baseline=True,
                        clip_during_training=False,
                    )
                )

    # A few longer, lower-learning-rate candidates.
    for n_factors in [2, 5, 10]:
        for reg in [0.05, 0.1, 0.2]:
            rows.append(
                dict(
                    n_factors=n_factors,
                    n_epochs=60,
                    learning_rate=0.001,
                    reg=reg,
                    reg_bias=0.05,
                    init_std=0.005,
                    init_with_baseline=True,
                    clip_during_training=False,
                )
            )

    # Keep the previous default as a comparison point.
    rows.append(
        dict(
            n_factors=20,
            n_epochs=25,
            learning_rate=0.01,
            reg=0.02,
            reg_bias=0.02,
            init_std=0.1,
            init_with_baseline=False,
            clip_during_training=False,
        )
    )
    return rows


def write_csv(path: Path, rows: List[Dict[str, float | int | bool | str]]) -> None:
    fieldnames = [
        "status",
        "n_factors",
        "n_epochs",
        "learning_rate",
        "reg",
        "reg_bias",
        "init_std",
        "init_with_baseline",
        "clip_during_training",
        "rmse",
        "std_rmse",
        "train_seconds",
        "predict_seconds",
        "memory_mb",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fixed_split_search(full_train: Sequence[TrainRecord]) -> List[Dict[str, float | int | bool | str]]:
    train_data, valid_data = train_valid_split(full_train, valid_ratio=0.2, seed=SEED)
    rows: List[Dict[str, float | int | bool | str]] = []
    configs = config_rows()

    for index, config in enumerate(configs, start=1):
        row: Dict[str, float | int | bool | str] = {**config}
        try:
            metrics = evaluate(make_model(config), train_data, valid_data)
            row.update({"status": "ok", "std_rmse": 0.0, "error": "", **metrics})
        except Exception as exc:  # noqa: BLE001
            row.update(
                {
                    "status": "failed",
                    "rmse": float("inf"),
                    "std_rmse": 0.0,
                    "train_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "memory_mb": 0.0,
                    "error": type(exc).__name__,
                }
            )
        rows.append(row)
        best = min((r for r in rows if r["status"] == "ok"), key=lambda r: float(r["rmse"]), default=None)
        if best is not None:
            print(
                f"[{index:03d}/{len(configs)}] rmse={float(row['rmse']):.6f} "
                f"best={float(best['rmse']):.6f} factors={best['n_factors']} "
                f"lr={best['learning_rate']} reg={best['reg']} epochs={best['n_epochs']}"
            )

    return rows


def cross_validate(full_train: Sequence[TrainRecord], configs: Sequence[Dict[str, float | int | bool | str]]) -> List[Dict[str, float | int | bool | str]]:
    folds = build_user_k_folds(full_train, n_splits=5, seed=SEED)
    rows: List[Dict[str, float | int | bool | str]] = []

    for config in configs:
        rmses: List[float] = []
        train_seconds: List[float] = []
        predict_seconds: List[float] = []
        memory_mb: List[float] = []
        status = "ok"
        error = ""

        for valid_index in range(5):
            valid_data = folds[valid_index]
            train_data = [
                record
                for fold_index, fold in enumerate(folds)
                if fold_index != valid_index
                for record in fold
            ]
            try:
                metrics = evaluate(make_model(config), train_data, valid_data)
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = type(exc).__name__
                break
            rmses.append(metrics["rmse"])
            train_seconds.append(metrics["train_seconds"])
            predict_seconds.append(metrics["predict_seconds"])
            memory_mb.append(metrics["memory_mb"])

        row: Dict[str, float | int | bool | str] = {
            **config,
            "status": status,
            "rmse": mean(rmses) if rmses else float("inf"),
            "std_rmse": std(rmses),
            "train_seconds": mean(train_seconds),
            "predict_seconds": mean(predict_seconds),
            "memory_mb": mean(memory_mb),
            "error": error,
        }
        rows.append(row)
        print(
            f"CV {status}: rmse={float(row['rmse']):.6f} std={float(row['std_rmse']):.6f} "
            f"factors={row['n_factors']} lr={row['learning_rate']} reg={row['reg']} epochs={row['n_epochs']}"
        )

    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_train = read_train(str(DATA_DIR / "Train.txt"))

    fixed_rows = fixed_split_search(full_train)
    fixed_path = RESULTS_DIR / "mf_quick_tuning_fixed.csv"
    write_csv(fixed_path, fixed_rows)

    top_configs = sorted(
        [row for row in fixed_rows if row["status"] == "ok"],
        key=lambda row: float(row["rmse"]),
    )[:5]
    cv_rows = cross_validate(full_train, top_configs)
    cv_path = RESULTS_DIR / "mf_quick_tuning_cv.csv"
    write_csv(cv_path, cv_rows)

    print(f"Saved fixed search to: {fixed_path}")
    print(f"Saved CV search to: {cv_path}")


if __name__ == "__main__":
    main()

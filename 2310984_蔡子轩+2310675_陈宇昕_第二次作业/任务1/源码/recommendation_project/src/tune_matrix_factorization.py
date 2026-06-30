"""Tune matrix factorization hyperparameters on Train.txt only."""

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

    train_start = time.perf_counter()
    with np.errstate(over="raise", invalid="raise"):
        model.fit(train_data)
    train_seconds = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    predictions = model.batch_predict(valid_pairs)
    predict_seconds = time.perf_counter() - predict_start

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


def make_model(config: Dict[str, float | int]) -> MatrixFactorizationModel:
    return MatrixFactorizationModel(
        n_factors=int(config["n_factors"]),
        n_epochs=int(config["n_epochs"]),
        learning_rate=float(config["learning_rate"]),
        reg=float(config["reg"]),
        reg_bias=float(config["reg_bias"]),
        init_std=float(config["init_std"]),
        seed=SEED,
    )


def write_rows(path: Path, rows: List[Dict[str, float | int | str]]) -> None:
    fieldnames = [
        "status",
        "n_factors",
        "n_epochs",
        "learning_rate",
        "reg",
        "reg_bias",
        "init_std",
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


def search_fixed_split(full_train: Sequence[TrainRecord]) -> List[Dict[str, float | int | str]]:
    train_data, valid_data = train_valid_split(full_train, valid_ratio=0.2, seed=SEED)
    configs = [
        # Original implementation and simple nearby variants.
        dict(n_factors=20, n_epochs=25, learning_rate=0.01, reg=0.02, reg_bias=0.02, init_std=0.1),
        dict(n_factors=5, n_epochs=25, learning_rate=0.01, reg=0.02, reg_bias=0.02, init_std=0.1),
        dict(n_factors=2, n_epochs=25, learning_rate=0.01, reg=0.02, reg_bias=0.02, init_std=0.1),
        # Smaller initialization keeps the dot-product term from dominating early updates.
        dict(n_factors=2, n_epochs=25, learning_rate=0.005, reg=0.02, reg_bias=0.02, init_std=0.01),
        dict(n_factors=5, n_epochs=25, learning_rate=0.005, reg=0.02, reg_bias=0.02, init_std=0.01),
        dict(n_factors=10, n_epochs=25, learning_rate=0.005, reg=0.02, reg_bias=0.02, init_std=0.01),
        dict(n_factors=2, n_epochs=40, learning_rate=0.005, reg=0.02, reg_bias=0.02, init_std=0.01),
        dict(n_factors=5, n_epochs=40, learning_rate=0.005, reg=0.02, reg_bias=0.02, init_std=0.01),
        # Lower learning rates for stability.
        dict(n_factors=2, n_epochs=40, learning_rate=0.002, reg=0.02, reg_bias=0.02, init_std=0.01),
        dict(n_factors=5, n_epochs=40, learning_rate=0.002, reg=0.02, reg_bias=0.02, init_std=0.01),
        dict(n_factors=10, n_epochs=40, learning_rate=0.002, reg=0.02, reg_bias=0.02, init_std=0.01),
        dict(n_factors=5, n_epochs=60, learning_rate=0.002, reg=0.02, reg_bias=0.02, init_std=0.01),
        dict(n_factors=10, n_epochs=60, learning_rate=0.002, reg=0.02, reg_bias=0.02, init_std=0.01),
        # Stronger regularization for sparse long-tail items.
        dict(n_factors=2, n_epochs=40, learning_rate=0.002, reg=0.05, reg_bias=0.02, init_std=0.01),
        dict(n_factors=5, n_epochs=40, learning_rate=0.002, reg=0.05, reg_bias=0.02, init_std=0.01),
        dict(n_factors=10, n_epochs=40, learning_rate=0.002, reg=0.05, reg_bias=0.02, init_std=0.01),
        dict(n_factors=5, n_epochs=60, learning_rate=0.002, reg=0.05, reg_bias=0.02, init_std=0.01),
        dict(n_factors=10, n_epochs=60, learning_rate=0.002, reg=0.05, reg_bias=0.02, init_std=0.01),
        dict(n_factors=5, n_epochs=60, learning_rate=0.001, reg=0.05, reg_bias=0.02, init_std=0.01),
        dict(n_factors=10, n_epochs=60, learning_rate=0.001, reg=0.05, reg_bias=0.02, init_std=0.01),
        dict(n_factors=5, n_epochs=80, learning_rate=0.001, reg=0.05, reg_bias=0.02, init_std=0.01),
        # Bias regularization variants.
        dict(n_factors=5, n_epochs=60, learning_rate=0.002, reg=0.05, reg_bias=0.05, init_std=0.01),
        dict(n_factors=10, n_epochs=60, learning_rate=0.002, reg=0.05, reg_bias=0.05, init_std=0.01),
        dict(n_factors=5, n_epochs=60, learning_rate=0.001, reg=0.05, reg_bias=0.05, init_std=0.01),
        dict(n_factors=5, n_epochs=80, learning_rate=0.001, reg=0.05, reg_bias=0.05, init_std=0.01),
        # Very small dot-product term, close to a bias-only MF.
        dict(n_factors=2, n_epochs=80, learning_rate=0.001, reg=0.1, reg_bias=0.02, init_std=0.005),
        dict(n_factors=5, n_epochs=80, learning_rate=0.001, reg=0.1, reg_bias=0.02, init_std=0.005),
        dict(n_factors=10, n_epochs=80, learning_rate=0.001, reg=0.1, reg_bias=0.02, init_std=0.005),
        dict(n_factors=5, n_epochs=100, learning_rate=0.001, reg=0.1, reg_bias=0.02, init_std=0.005),
    ]

    rows: List[Dict[str, float | int | str]] = []
    total = len(configs)
    for index, config in enumerate(configs, start=1):
        row: Dict[str, float | int | str] = {**config}
        try:
            metrics = evaluate(make_model(config), train_data, valid_data)
            row.update({"status": "ok", "std_rmse": 0.0, "error": "", **metrics})
        except Exception as exc:  # noqa: BLE001 - record failed hyperparameters.
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
        if index % 5 == 0 or index == total:
            best = min((r for r in rows if r["status"] == "ok"), key=lambda r: float(r["rmse"]), default=None)
            if best:
                print(
                    f"[{index}/{total}] best rmse={float(best['rmse']):.6f} "
                    f"config={best['n_factors']},{best['n_epochs']},{best['learning_rate']},"
                    f"{best['reg']},{best['reg_bias']},{best['init_std']}"
                )
            else:
                print(f"[{index}/{total}] no valid configuration yet")
    return rows


def cross_validate_top(full_train: Sequence[TrainRecord], top_configs: Sequence[Dict[str, float | int | str]]) -> List[Dict[str, float | int | str]]:
    folds = build_user_k_folds(full_train, n_splits=5, seed=SEED)
    rows: List[Dict[str, float | int | str]] = []

    for config in top_configs:
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

        row: Dict[str, float | int | str] = {
            "status": status,
            "n_factors": int(config["n_factors"]),
            "n_epochs": int(config["n_epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "reg": float(config["reg"]),
            "reg_bias": float(config["reg_bias"]),
            "init_std": float(config["init_std"]),
            "rmse": mean(rmses) if rmses else float("inf"),
            "std_rmse": std(rmses),
            "train_seconds": mean(train_seconds),
            "predict_seconds": mean(predict_seconds),
            "memory_mb": mean(memory_mb),
            "error": error,
        }
        rows.append(row)
        print(
            f"cv {row['status']} rmse={float(row['rmse']):.6f} std={float(row['std_rmse']):.6f} "
            f"config={row['n_factors']},{row['n_epochs']},{row['learning_rate']},"
            f"{row['reg']},{row['reg_bias']},{row['init_std']}"
        )

    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_train = read_train(str(DATA_DIR / "Train.txt"))

    fixed_rows = search_fixed_split(full_train)
    fixed_path = RESULTS_DIR / "mf_tuning_fixed_split.csv"
    write_rows(fixed_path, fixed_rows)

    top_rows = sorted(
        [row for row in fixed_rows if row["status"] == "ok"],
        key=lambda row: float(row["rmse"]),
    )[:8]
    cv_rows = cross_validate_top(full_train, top_rows)
    cv_path = RESULTS_DIR / "mf_tuning_cv_top.csv"
    write_rows(cv_path, cv_rows)

    print(f"Saved fixed-split tuning to: {fixed_path}")
    print(f"Saved top-config CV tuning to: {cv_path}")


if __name__ == "__main__":
    main()

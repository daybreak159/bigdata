"""Run parameter sensitivity experiments on an internal validation split."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from utils import TrainRecord, clip_rating, read_train, rmse, train_valid_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 2026


def evaluate_model(model, train_data: Sequence[TrainRecord], valid_data: Sequence[TrainRecord]):
    valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
    y_true = [rating for _, _, rating in valid_data]

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

    return {
        "rmse": rmse(y_true, y_pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "memory_mb": model.approximate_size_bytes() / (1024 * 1024),
    }


def run_itemcf_sweep(
    train_data: Sequence[TrainRecord],
    valid_data: Sequence[TrainRecord],
) -> List[Dict[str, float | int | str]]:
    base_params: Dict[str, int | float] = dict(
        top_k=20,
        similarity_top_n=80,
        shrinkage=10.0,
        min_common=2,
        block_size=256,
    )
    sweeps: List[Tuple[str, str, List[int | float]]] = [
        ("shrinkage", "shrinkage", [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0]),
        ("min_common", "min_common", [1, 2, 3, 4, 5]),
        ("top_k", "top_k", [5, 10, 20, 30, 40, 60, 80]),
        ("similarity_top_n", "similarity_top_n", [40, 60, 80, 100, 120, 160, 200]),
        ("block_size", "block_size", [64, 128, 256, 512, 1024]),
    ]

    configs: List[Tuple[str, str, int | float | str, Dict[str, int | float]]] = []
    for group, parameter_name, values in sweeps:
        for value in values:
            params = dict(base_params)
            params[parameter_name] = value
            configs.append((group, parameter_name, value, params))

    joint_params = dict(base_params)
    joint_params.update(top_k=40, similarity_top_n=120)
    configs.append(("joint", "top_k+similarity_top_n", "40+120", joint_params))

    final_joint_params = dict(base_params)
    final_joint_params.update(top_k=40, similarity_top_n=200, shrinkage=50.0)
    configs.append(("joint", "top_k+similarity_top_n+shrinkage", "40+200+50", final_joint_params))

    rows: List[Dict[str, float | int | str]] = []
    for group, parameter_name, parameter_value, params in configs:
        name = f"{group}_{parameter_value}"
        model = ItemCFModel(seed=SEED, **params)
        metrics = evaluate_model(model, train_data, valid_data)
        rows.append(
            {
                "model": "ItemCF",
                "config": name,
                "sweep_group": group,
                "changed_parameter": parameter_name,
                "parameter_value": parameter_value,
                **params,
                **metrics,
            }
        )
    return rows


def run_mf_sweep(
    train_data: Sequence[TrainRecord],
    valid_data: Sequence[TrainRecord],
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for n_factors in [5, 10, 20, 40, 80]:
        model = MatrixFactorizationModel(
            n_factors=n_factors,
            n_epochs=25,
            learning_rate=0.01,
            reg=0.02,
            reg_bias=0.02,
            seed=SEED,
        )
        metrics = evaluate_model(model, train_data, valid_data)
        rows.append(
            {
                "model": "MatrixFactorization",
                "config": f"n_factors_{n_factors}",
                "n_factors": n_factors,
                **metrics,
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, float | int | str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_train = read_train(str(DATA_DIR / "Train.txt"))
    train_data, valid_data = train_valid_split(full_train, valid_ratio=0.2, seed=SEED)

    print(f"Train records: {len(train_data)}")
    print(f"Valid records: {len(valid_data)}")

    itemcf_rows = run_itemcf_sweep(train_data, valid_data)
    itemcf_path = RESULTS_DIR / "itemcf_parameter_results.csv"
    write_csv(itemcf_path, itemcf_rows)

    print(f"Saved ItemCF parameter results to: {itemcf_path}")


if __name__ == "__main__":
    main()

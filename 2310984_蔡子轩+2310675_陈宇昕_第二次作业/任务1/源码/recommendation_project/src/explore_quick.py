"""Quick, incremental exploration that appends each finished result to CSV."""

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
from utils import TrainRecord, clip_rating, read_train, rmse, train_valid_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 2026


def evaluate(
    name: str,
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

    y_pred = [
        clip_rating(score, model.min_rating, model.max_rating)
        for _, _, score in predictions
    ]
    return {
        "name": name,
        "status": "ok",
        "rmse": rmse(y_true, y_pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "memory_mb": model.approximate_size_bytes() / (1024 * 1024),
        "error": "",
    }


def configs() -> list[tuple[str, Callable[[], object]]]:
    items: list[tuple[str, Callable[[], object]]] = []

    for reg_user in [6.0, 8.0, 10.0, 12.0]:
        for reg_item in [10.0, 12.0, 15.0, 18.0]:
            name = f"Baseline_ru{reg_user:g}_ri{reg_item:g}"
            items.append(
                (
                    name,
                    lambda ru=reg_user, ri=reg_item: BaselineModel(
                        reg_user=ru,
                        reg_item=ri,
                        n_iters=15,
                        seed=SEED,
                    ),
                )
            )

    for top_k in [40, 60, 80]:
        for top_n in [200, 240, 320]:
            for shrinkage in [50.0, 80.0, 120.0]:
                name = f"ItemCF_k{top_k}_n{top_n}_sh{shrinkage:g}_mc2"
                items.append(
                    (
                        name,
                        lambda k=top_k, n=top_n, sh=shrinkage: ItemCFModel(
                            top_k=k,
                            similarity_top_n=n,
                            shrinkage=sh,
                            min_common=2,
                            block_size=512,
                            seed=SEED,
                        ),
                    )
                )

    for top_k in [20, 40, 80]:
        for top_n in [120, 200, 400]:
            for shrinkage in [30.0, 50.0, 80.0]:
                for min_common in [2, 3, 5]:
                    if top_k > top_n:
                        continue
                    name = f"UserCF_k{top_k}_n{top_n}_sh{shrinkage:g}_mc{min_common}"
                    items.append(
                        (
                            name,
                            lambda k=top_k, n=top_n, sh=shrinkage, mc=min_common: UserCFModel(
                                top_k=k,
                                similarity_top_n=n,
                                shrinkage=sh,
                                min_common=mc,
                                seed=SEED,
                            ),
                        )
                    )

    for n_factors in [3, 4, 5, 6, 8]:
        for learning_rate in [0.0015, 0.002, 0.0025]:
            for reg in [0.15, 0.2, 0.3]:
                name = f"MF_f{n_factors}_lr{learning_rate:g}_reg{reg:g}"
                items.append(
                    (
                        name,
                        lambda f=n_factors, lr=learning_rate, rg=reg: MatrixFactorizationModel(
                            n_factors=f,
                            n_epochs=12,
                            learning_rate=lr,
                            reg=rg,
                            reg_bias=0.05,
                            init_std=0.01,
                            init_with_baseline=True,
                            freeze_bias=True,
                            seed=SEED,
                        ),
                    )
                )

    return items


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "exploration_quick_fixed_split.csv"
    full_train = read_train(str(DATA_DIR / "Train.txt"))
    train_data, valid_data = train_valid_split(full_train, valid_ratio=0.2, seed=SEED)
    all_configs = configs()
    fieldnames = [
        "name",
        "status",
        "rmse",
        "train_seconds",
        "predict_seconds",
        "memory_mb",
        "error",
    ]

    best_name = ""
    best_rmse = math.inf
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, (name, builder) in enumerate(all_configs, start=1):
            try:
                row = evaluate(name, builder, train_data, valid_data)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "name": name,
                    "status": "failed",
                    "rmse": math.inf,
                    "train_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "memory_mb": 0.0,
                    "error": type(exc).__name__,
                }
            writer.writerow(row)
            f.flush()

            current_rmse = float(row["rmse"])
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_name = name
            print(
                f"[{index:03d}/{len(all_configs)}] {name} "
                f"rmse={current_rmse:.6f} best={best_name}:{best_rmse:.6f}",
                flush=True,
            )

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

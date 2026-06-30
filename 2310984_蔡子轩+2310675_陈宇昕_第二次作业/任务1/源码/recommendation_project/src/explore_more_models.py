"""Additional offline exploration for recommendation models.

This script intentionally writes to exploration_* CSV files so it does not
overwrite the report's official experiment results.
"""

from __future__ import annotations

import csv
import itertools
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

from baseline import BaselineModel
from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from user_cf import UserCFModel
from utils import (
    Prediction,
    TrainRecord,
    build_user_k_folds,
    clip_rating,
    read_train,
    rmse,
    train_valid_split,
)


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


def evaluate_predictions(
    valid_data: Sequence[TrainRecord],
    predictions: Sequence[Prediction],
    min_rating: float,
    max_rating: float,
) -> float:
    y_true = [rating for _, _, rating in valid_data]
    y_pred = [clip_rating(score, min_rating, max_rating) for _, _, score in predictions]
    return rmse(y_true, y_pred)


def evaluate_model(
    name: str,
    build_model: Callable[[], object],
    train_data: Sequence[TrainRecord],
    valid_data: Sequence[TrainRecord],
) -> Dict[str, float | str]:
    valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
    model = build_model()

    train_start = time.perf_counter()
    model.fit(train_data)
    train_seconds = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    predictions = model.batch_predict(valid_pairs)
    predict_seconds = time.perf_counter() - predict_start

    return {
        "name": name,
        "rmse": evaluate_predictions(
            valid_data,
            predictions,
            model.min_rating,
            model.max_rating,
        ),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "memory_mb": model.approximate_size_bytes() / (1024 * 1024),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_builders() -> List[tuple[str, Callable[[], object]]]:
    builders: List[tuple[str, Callable[[], object]]] = []

    for reg_user, reg_item, n_iters in itertools.product(
        [5.0, 8.0, 10.0, 15.0],
        [8.0, 10.0, 15.0, 20.0],
        [10, 15],
    ):
        name = f"Baseline_ru{reg_user:g}_ri{reg_item:g}_it{n_iters}"
        builders.append(
            (
                name,
                lambda ru=reg_user, ri=reg_item, it=n_iters: BaselineModel(
                    reg_user=ru,
                    reg_item=ri,
                    n_iters=it,
                    seed=SEED,
                ),
            )
        )

    for top_k, top_n, shrinkage, min_common in itertools.product(
        [40, 60],
        [200, 240, 320],
        [50.0, 80.0, 120.0],
        [1, 2, 3],
    ):
        if top_k > top_n:
            continue
        name = f"ItemCF_k{top_k}_n{top_n}_sh{shrinkage:g}_mc{min_common}"
        builders.append(
            (
                name,
                lambda k=top_k, n=top_n, sh=shrinkage, mc=min_common: ItemCFModel(
                    top_k=k,
                    similarity_top_n=n,
                    shrinkage=sh,
                    min_common=mc,
                    block_size=512,
                    seed=SEED,
                ),
            )
        )

    for top_k, top_n, shrinkage, min_common in itertools.product(
        [20, 40, 80, 120],
        [120, 200, 400],
        [30.0, 50.0, 80.0],
        [2, 3, 5],
    ):
        if top_k > top_n:
            continue
        name = f"UserCF_k{top_k}_n{top_n}_sh{shrinkage:g}_mc{min_common}"
        builders.append(
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

    mf_configs = []
    for n_factors in [3, 4, 5, 6, 8]:
        for learning_rate in [0.0015, 0.002, 0.0025]:
            for reg in [0.15, 0.2, 0.3]:
                mf_configs.append((n_factors, 12, learning_rate, reg))
    for n_factors, n_epochs, learning_rate, reg in mf_configs:
        name = f"MF_f{n_factors}_ep{n_epochs}_lr{learning_rate:g}_reg{reg:g}"
        builders.append(
            (
                name,
                lambda f=n_factors, ep=n_epochs, lr=learning_rate, rg=reg: MatrixFactorizationModel(
                    n_factors=f,
                    n_epochs=ep,
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

    return builders


def fixed_split_screen(full_train: Sequence[TrainRecord]) -> list[dict[str, object]]:
    train_data, valid_data = train_valid_split(full_train, valid_ratio=0.2, seed=SEED)
    rows: list[dict[str, object]] = []
    builders = model_builders()
    total = len(builders)

    for index, (name, builder) in enumerate(builders, start=1):
        try:
            row = evaluate_model(name, builder, train_data, valid_data)
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - exploration should continue.
            row = {
                "name": name,
                "status": "failed",
                "rmse": float("inf"),
                "train_seconds": 0.0,
                "predict_seconds": 0.0,
                "memory_mb": 0.0,
                "error": type(exc).__name__,
            }
        rows.append(row)
        if index % 25 == 0 or index == total:
            best = min(
                (row for row in rows if row["status"] == "ok"),
                key=lambda row: float(row["rmse"]),
                default=None,
            )
            if best:
                print(f"[screen {index}/{total}] best {best['name']} rmse={float(best['rmse']):.6f}")

    return rows


def build_named_model(name: str):
    for candidate_name, builder in model_builders():
        if candidate_name == name:
            return builder()
    raise ValueError(f"unknown model name: {name}")


def cross_validate_model_names(
    full_train: Sequence[TrainRecord],
    names: Sequence[str],
) -> list[dict[str, object]]:
    folds = build_user_k_folds(full_train, n_splits=5, seed=SEED)
    rows: list[dict[str, object]] = []

    for name in names:
        fold_rmses: list[float] = []
        train_seconds: list[float] = []
        predict_seconds: list[float] = []
        memory_mb: list[float] = []

        for valid_index in range(5):
            valid_data = folds[valid_index]
            train_data = [
                record
                for fold_index, fold in enumerate(folds)
                if fold_index != valid_index
                for record in fold
            ]
            row = evaluate_model(name, lambda n=name: build_named_model(n), train_data, valid_data)
            fold_rmses.append(float(row["rmse"]))
            train_seconds.append(float(row["train_seconds"]))
            predict_seconds.append(float(row["predict_seconds"]))
            memory_mb.append(float(row["memory_mb"]))

        result: dict[str, object] = {
            "name": name,
            "mean_rmse": mean(fold_rmses),
            "std_rmse": std(fold_rmses),
            "avg_train_seconds": mean(train_seconds),
            "avg_predict_seconds": mean(predict_seconds),
            "avg_memory_mb": mean(memory_mb),
            "status": "ok",
        }
        for idx, fold_rmse in enumerate(fold_rmses, start=1):
            result[f"fold_{idx}_rmse"] = fold_rmse
        rows.append(result)
        print(f"[cv] {name} mean={result['mean_rmse']:.6f} std={result['std_rmse']:.6f}")

    return rows


def cache_fold_predictions(
    full_train: Sequence[TrainRecord],
    names: Sequence[str],
) -> tuple[list[list[float]], list[list[list[float]]], float, float]:
    folds = build_user_k_folds(full_train, n_splits=5, seed=SEED)
    y_true_by_fold: list[list[float]] = []
    predictions_by_model: list[list[list[float]]] = [[] for _ in names]
    rating_min = 10.0
    rating_max = 100.0

    for valid_index in range(5):
        valid_data = folds[valid_index]
        train_data = [
            record
            for fold_index, fold in enumerate(folds)
            if fold_index != valid_index
            for record in fold
        ]
        valid_pairs = [(user_id, item_id) for user_id, item_id, _ in valid_data]
        y_true_by_fold.append([rating for _, _, rating in valid_data])

        for model_index, name in enumerate(names):
            model = build_named_model(name)
            model.fit(train_data)
            preds = [score for _, _, score in model.batch_predict(valid_pairs)]
            predictions_by_model[model_index].append(preds)
            rating_min = model.min_rating
            rating_max = model.max_rating
        print(f"[cache] fold {valid_index + 1}/5")

    return y_true_by_fold, predictions_by_model, rating_min, rating_max


def weight_grid(step: float, n_models: int) -> Iterable[tuple[float, ...]]:
    slots = int(round(1.0 / step))
    if n_models == 2:
        for a in range(slots + 1):
            yield (a * step, (slots - a) * step)
        return
    if n_models == 3:
        for a in range(slots + 1):
            for b in range(slots - a + 1):
                c = slots - a - b
                yield (a * step, b * step, c * step)
        return
    raise ValueError("only 2 or 3 models are supported")


def evaluate_weighted_cached(
    y_true_by_fold: Sequence[Sequence[float]],
    predictions_by_model: Sequence[Sequence[Sequence[float]]],
    weights: Sequence[float],
    rating_min: float,
    rating_max: float,
) -> tuple[float, float]:
    fold_rmses: list[float] = []
    for fold_index, y_true in enumerate(y_true_by_fold):
        blended = []
        n_records = len(y_true)
        for record_index in range(n_records):
            score = sum(
                weight * predictions_by_model[model_index][fold_index][record_index]
                for model_index, weight in enumerate(weights)
            )
            blended.append(clip_rating(score, rating_min, rating_max))
        fold_rmses.append(rmse(y_true, blended))
    return mean(fold_rmses), std(fold_rmses)


def blend_search(
    full_train: Sequence[TrainRecord],
    names: Sequence[str],
    step: float = 0.02,
) -> list[dict[str, object]]:
    y_true_by_fold, predictions_by_model, rating_min, rating_max = cache_fold_predictions(full_train, names)
    rows: list[dict[str, object]] = []
    for weights in weight_grid(step, len(names)):
        mean_rmse, std_rmse = evaluate_weighted_cached(
            y_true_by_fold,
            predictions_by_model,
            weights,
            rating_min,
            rating_max,
        )
        row: dict[str, object] = {
            "models": "+".join(names),
            "weights": ";".join(f"{weight:.2f}" for weight in weights),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
        }
        for model_name, weight in zip(names, weights):
            row[f"weight_{model_name}"] = weight
        rows.append(row)
    rows.sort(key=lambda row: float(row["mean_rmse"]))
    best = rows[0]
    print(f"[blend] {best['models']} {best['weights']} mean={float(best['mean_rmse']):.6f}")
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_train = read_train(str(DATA_DIR / "Train.txt"))

    screen_rows = fixed_split_screen(full_train)
    write_csv(RESULTS_DIR / "exploration_fixed_split_screen.csv", screen_rows)

    ok_rows = [row for row in screen_rows if row["status"] == "ok"]
    top_names = [
        str(row["name"])
        for row in sorted(ok_rows, key=lambda row: float(row["rmse"]))[:18]
    ]
    official_names = [
        "Baseline_ru10_ri15_it15",
        "ItemCF_k40_n200_sh50_mc2",
        "MF_f4_ep12_lr0.002_reg0.2",
    ]
    candidate_names = sorted(set(top_names + official_names))
    print("CV candidates:")
    for name in candidate_names:
        print(f"  {name}")

    cv_rows = cross_validate_model_names(full_train, candidate_names)
    write_csv(RESULTS_DIR / "exploration_cv_candidates.csv", cv_rows)

    best_cv_names = [
        str(row["name"])
        for row in sorted(cv_rows, key=lambda row: float(row["mean_rmse"]))[:6]
    ]
    print("Blend candidates:")
    for name in best_cv_names:
        print(f"  {name}")

    blend_rows: list[dict[str, object]] = []
    # Pair and triple blends among the strongest diverse candidates.
    for names in itertools.combinations(best_cv_names[:5], 2):
        blend_rows.extend(blend_search(full_train, names, step=0.01)[:20])
    for names in itertools.combinations(best_cv_names[:5], 3):
        blend_rows.extend(blend_search(full_train, names, step=0.02)[:20])
    blend_rows.sort(key=lambda row: float(row["mean_rmse"]))
    write_csv(RESULTS_DIR / "exploration_blend_search.csv", blend_rows)

    print("Top exploration results:")
    for row in blend_rows[:10]:
        print(row)


if __name__ == "__main__":
    main()

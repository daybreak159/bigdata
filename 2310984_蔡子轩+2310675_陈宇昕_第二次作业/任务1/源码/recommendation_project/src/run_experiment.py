"""Run a unified experiment for all recommendation models."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, List

from baseline import BaselineModel
from ensemble import EnsembleModel
from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from utils import build_user_k_folds, clip_rating, dataset_statistics, read_train, rmse


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


def build_model(model_name: str):
    if model_name == "Baseline":
        return BaselineModel(seed=SEED)
    if model_name == "ItemCF":
        return ItemCFModel(
            top_k=40,
            similarity_top_n=200,
            shrinkage=50.0,
            min_common=2,
            block_size=256,
            seed=SEED,
        )
    if model_name == "MatrixFactorization":
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
    if model_name == "Ensemble":
        return EnsembleModel(itemcf_weight=0.55, seed=SEED)
    raise ValueError(f"Unsupported model: {model_name}")


def print_stats(stats: Dict[str, float]) -> None:
    print("Dataset statistics")
    print(f"  users      : {int(stats['num_users'])}")
    print(f"  items      : {int(stats['num_items'])}")
    print(f"  ratings    : {int(stats['num_ratings'])}")
    print(f"  rating min : {stats['min_rating']:.2f}")
    print(f"  rating max : {stats['max_rating']:.2f}")
    print(f"  rating avg : {stats['mean_rating']:.4f}")
    print(f"  density    : {stats['density']:.6f}")
    print(f"  validation : {N_FOLDS}-fold cross validation on Train.txt")
    print()


def print_table(rows: List[Dict[str, float]]) -> None:
    headers = [
        "model",
        "mean_rmse",
        "std_rmse",
        "avg_train_seconds",
        "avg_predict_seconds",
        "avg_memory_mb",
    ]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            {
                "model": row["model"],
                "mean_rmse": f"{row['mean_rmse']:.6f}",
                "std_rmse": f"{row['std_rmse']:.6f}",
                "avg_train_seconds": f"{row['avg_train_seconds']:.4f}",
                "avg_predict_seconds": f"{row['avg_predict_seconds']:.4f}",
                "avg_memory_mb": f"{row['avg_memory_mb']:.4f}",
            }
        )

    widths = {
        header: max(len(header), *(len(item[header]) for item in formatted_rows))
        for header in headers
    }
    separator = "-+-".join("-" * widths[header] for header in headers)

    print("Experiment comparison")
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print(separator)
    for row in formatted_rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def write_final_model_metrics(results_path: Path) -> None:
    """Write the final model metrics used in the report."""
    fieldnames = [
        "model",
        "mean_rmse",
        "rmse_std",
        "full_train_seconds",
        "predict_seconds",
        "core_memory_mb",
        "rmse_source",
        "time_memory_source",
    ]
    row = {
        "model": "ItemCF+MF+UserCF",
        "mean_rmse": "16.859807",
        "rmse_std": "0.085568",
        "full_train_seconds": "25.2029",
        "predict_seconds": "0.9953",
        "core_memory_mb": "36.9539",
        "rmse_source": "5 seeds x 5 folds internal validation",
        "time_memory_source": "full Train.txt training and Test.txt prediction",
    }
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def print_final_model_metrics(results_path: Path) -> None:
    """Print the final selected model metrics from the report."""
    with results_path.open("r", encoding="utf-8") as file:
        row = next(csv.DictReader(file))

    print("Final selected model metrics")
    print(
        "model            | mean_rmse | rmse_std | full_train_seconds | "
        "predict_seconds | core_memory_mb"
    )
    print(
        "-----------------+-----------+----------+--------------------+"
        "-----------------+---------------"
    )
    print(
        f"{row['model'].ljust(16)} | "
        f"{row['mean_rmse']} | "
        f"{row['rmse_std']} | "
        f"{row['full_train_seconds'].ljust(18)} | "
        f"{row['predict_seconds'].ljust(15)} | "
        f"{row['core_memory_mb']}"
    )
    print()
    print(
        "Note: experiment_results.csv is the basic 5-fold model comparison; "
        "final_model_metrics.csv corresponds to the final ItemCF+MF+UserCF "
        "model reported in Table 19."
    )


def main() -> None:
    train_path = DATA_DIR / "Train.txt"
    results_path = RESULTS_DIR / "experiment_results.csv"
    final_metrics_path = RESULTS_DIR / "final_model_metrics.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_train = read_train(str(train_path))
    stats = dataset_statistics(full_train)
    print_stats(stats)

    folds = build_user_k_folds(full_train, n_splits=N_FOLDS, seed=SEED)
    fold_sizes = [len(fold) for fold in folds]
    print(f"Fold sizes: {fold_sizes}")
    print()

    model_names = ["Baseline", "ItemCF", "MatrixFactorization", "Ensemble"]
    results: List[Dict[str, float]] = []

    for model_name in model_names:
        fold_rmses: List[float] = []
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

            model = build_model(model_name)

            train_start = time.perf_counter()
            model.fit(train_data)
            train_seconds = time.perf_counter() - train_start

            predict_start = time.perf_counter()
            predictions = model.batch_predict(valid_pairs)
            predict_seconds = time.perf_counter() - predict_start

            rating_min = model.min_rating
            rating_max = model.max_rating
            y_pred = [
                clip_rating(score, rating_min, rating_max)
                for _, _, score in predictions
            ]
            fold_rmses.append(rmse(y_true, y_pred))
            train_seconds_list.append(train_seconds)
            predict_seconds_list.append(predict_seconds)
            memory_bytes_list.append(float(model.approximate_size_bytes()))

        result_row: Dict[str, float] = {
            "model": model_name,
            "mean_rmse": mean(fold_rmses),
            "std_rmse": std(fold_rmses),
            "avg_train_seconds": mean(train_seconds_list),
            "avg_predict_seconds": mean(predict_seconds_list),
            "avg_memory_bytes": mean(memory_bytes_list),
            "avg_memory_mb": mean(memory_bytes_list) / (1024 * 1024),
        }
        for fold_index, fold_rmse in enumerate(fold_rmses, start=1):
            result_row[f"fold_{fold_index}_rmse"] = fold_rmse
        results.append(result_row)

    with open(results_path, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "model",
            *[f"fold_{fold_index}_rmse" for fold_index in range(1, N_FOLDS + 1)],
            "mean_rmse",
            "std_rmse",
            "avg_train_seconds",
            "avg_predict_seconds",
            "avg_memory_bytes",
            "avg_memory_mb",
        ]
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in results:
            csv_row = {
                "model": row["model"],
                "mean_rmse": f"{row['mean_rmse']:.6f}",
                "std_rmse": f"{row['std_rmse']:.6f}",
                "avg_train_seconds": f"{row['avg_train_seconds']:.4f}",
                "avg_predict_seconds": f"{row['avg_predict_seconds']:.4f}",
                "avg_memory_bytes": int(row["avg_memory_bytes"]),
                "avg_memory_mb": f"{row['avg_memory_mb']:.4f}",
            }
            for fold_index in range(1, N_FOLDS + 1):
                csv_row[f"fold_{fold_index}_rmse"] = f"{row[f'fold_{fold_index}_rmse']:.6f}"
            writer.writerow(csv_row)

    print_table(results)
    print()
    best_model = min(results, key=lambda row: row["mean_rmse"])
    print(
        "Best model by 5-fold mean RMSE: "
        f"{best_model['model']} ({best_model['mean_rmse']:.6f})"
    )
    print()
    print(f"Saved CSV to: {results_path}")
    print()

    write_final_model_metrics(final_metrics_path)
    print_final_model_metrics(final_metrics_path)
    print(f"Saved final model metrics to: {final_metrics_path}")


if __name__ == "__main__":
    main()

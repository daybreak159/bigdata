"""Compare compact-NumPy storage against previous nested-dict results.

Runs the same seeds / folds as validate_best_multi_seed.py but only for
the models that changed (ItemCF_tuned, UserCF_tuned, MF_tuned, triple ensemble).
Saves fold-level detail to compact_storage_comparison_folds.csv and a
summary to compact_storage_comparison.csv.

Previous baseline numbers are hard-coded from
exploration_best_methods_multiseed_summary.csv for easy diff printing.
"""

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

N_FOLDS = 5
SEEDS = [2024, 2025, 2026, 2027, 2028]

# Previous nested-dict results for diff output. These are historical comparison
# values only; current report metrics are written to compact_storage_comparison.csv.
PREVIOUS_NESTED_DICT = {
    "ItemCF_tuned_baseline": {"mean_rmse": 17.044970, "mean_memory_mb": 121.3278, "mean_predict_seconds": 1.2084},
    "UserCF_tuned":          {"mean_rmse": 17.541537, "mean_memory_mb": 26.6007,  "mean_predict_seconds": 0.4262},
    "MF_tuned":              {"mean_rmse": 17.137423, "mean_memory_mb": 1.0899,   "mean_predict_seconds": 0.0391},
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def fit_predict(
    builder: Callable[[], object],
    train_data: Sequence[TrainRecord],
    valid_data: Sequence[TrainRecord],
) -> dict:
    valid_pairs = [(u, i) for u, i, _ in valid_data]
    y_true = [r for _, _, r in valid_data]
    model = builder()

    t0 = time.perf_counter()
    model.fit(train_data)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.batch_predict(valid_pairs)
    predict_s = time.perf_counter() - t0

    scores = [s for _, _, s in preds]
    clipped = [clip_rating(s, model.min_rating, model.max_rating) for s in scores]
    return {
        "scores": scores,
        "rmse": rmse(y_true, clipped),
        "train_s": train_s,
        "predict_s": predict_s,
        "memory_mb": model.approximate_size_bytes() / (1024 * 1024),
        "min_rating": model.min_rating,
        "max_rating": model.max_rating,
    }


def blend_rmse(
    y_true: list[float],
    score_lists: list[list[float]],
    weights: tuple[float, ...],
    rmin: float,
    rmax: float,
) -> float:
    return rmse(
        y_true,
        [clip_rating(sum(w * s for w, s in zip(weights, row)), rmin, rmax)
         for row in zip(*score_lists)],
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    folds_path = RESULTS_DIR / "compact_storage_comparison_folds.csv"
    summary_path = RESULTS_DIR / "compact_storage_comparison.csv"

    full_train = read_train(str(DATA_DIR / "Train.txt"))

    fieldnames = ["seed", "fold", "model", "rmse", "train_s", "predict_s", "memory_mb"]
    all_rows: list[dict] = []

    with folds_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in SEEDS:
            folds = build_user_k_folds(full_train, n_splits=N_FOLDS, seed=seed)
            for vi in range(N_FOLDS):
                fold_num = vi + 1
                valid_data = folds[vi]
                train_data = [r for fi, fold in enumerate(folds) if fi != vi for r in fold]
                y_true = [r for _, _, r in valid_data]
                print(f"seed={seed} fold={fold_num}/{N_FOLDS}", flush=True)

                configs = [
                    ("ItemCF_tuned_baseline", lambda: ItemCFModel(
                        top_k=40, similarity_top_n=320, shrinkage=120.0,
                        min_common=2, block_size=512,
                        baseline_reg_user=6.0, baseline_reg_item=10.0, seed=seed,
                    )),
                    ("MF_tuned", lambda: MatrixFactorizationModel(
                        n_factors=48, n_epochs=12, learning_rate=0.002,
                        reg=0.3, reg_bias=0.05, init_std=0.01,
                        init_with_baseline=True, freeze_bias=True, seed=seed,
                    )),
                    ("UserCF_tuned", lambda: UserCFModel(
                        top_k=40, similarity_top_n=400, shrinkage=30.0,
                        min_common=2, seed=seed,
                    )),
                ]

                outputs: dict[str, dict] = {}
                for name, builder in configs:
                    out = fit_predict(builder, train_data, valid_data)
                    outputs[name] = out
                    row = {"seed": seed, "fold": fold_num, "model": name,
                           "rmse": f"{out['rmse']:.6f}", "train_s": f"{out['train_s']:.4f}",
                           "predict_s": f"{out['predict_s']:.4f}", "memory_mb": f"{out['memory_mb']:.4f}"}
                    writer.writerow(row)
                    all_rows.append(row)
                    print(f"  {name}: rmse={out['rmse']:.6f}  mem={out['memory_mb']:.2f}MB", flush=True)

                # Triple ensemble
                icf, mf, ucf = outputs["ItemCF_tuned_baseline"], outputs["MF_tuned"], outputs["UserCF_tuned"]
                ens_rmse = blend_rmse(
                    y_true,
                    [icf["scores"], mf["scores"], ucf["scores"]],
                    (0.43, 0.45, 0.12),
                    min(icf["min_rating"], mf["min_rating"], ucf["min_rating"]),
                    max(icf["max_rating"], mf["max_rating"], ucf["max_rating"]),
                )
                ens_row = {
                    "seed": seed, "fold": fold_num, "model": "Ensemble_triple_tuned",
                    "rmse": f"{ens_rmse:.6f}",
                    "train_s": f"{icf['train_s'] + mf['train_s'] + ucf['train_s']:.4f}",
                    "predict_s": f"{icf['predict_s'] + mf['predict_s'] + ucf['predict_s']:.4f}",
                    "memory_mb": f"{icf['memory_mb'] + mf['memory_mb'] + ucf['memory_mb']:.4f}",
                }
                writer.writerow(ens_row)
                all_rows.append(ens_row)
                print(f"  Ensemble_triple_tuned: rmse={ens_rmse:.6f}", flush=True)
                f.flush()

    # Write summary
    by_model: dict[str, list[dict]] = {}
    for row in all_rows:
        by_model.setdefault(row["model"], []).append(row)

    summary_fields = ["model", "mean_rmse", "std_rmse", "mean_train_s",
                      "mean_predict_s", "mean_memory_mb",
                      "rmse_diff_vs_previous", "memory_diff_vs_previous_mb"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for model_name, rows in by_model.items():
            rmses   = [float(r["rmse"]) for r in rows]
            trains  = [float(r["train_s"]) for r in rows]
            preds   = [float(r["predict_s"]) for r in rows]
            mems    = [float(r["memory_mb"]) for r in rows]
            prev    = PREVIOUS_NESTED_DICT.get(model_name, {})
            writer.writerow({
                "model": model_name,
                "mean_rmse": f"{_mean(rmses):.6f}",
                "std_rmse": f"{_std(rmses):.6f}",
                "mean_train_s": f"{_mean(trains):.4f}",
                "mean_predict_s": f"{_mean(preds):.4f}",
                "mean_memory_mb": f"{_mean(mems):.4f}",
                "rmse_diff_vs_previous": f"{_mean(rmses) - prev.get('mean_rmse', float('nan')):.6f}" if prev else "N/A",
                "memory_diff_vs_previous_mb": f"{_mean(mems) - prev.get('mean_memory_mb', float('nan')):.4f}" if prev else "N/A",
            })

    # Console diff summary
    print("\n=== Comparison vs previous ===")
    print(f"{'Model':<30} {'Old RMSE':>10} {'New RMSE':>10} {'ΔRMSE':>8} "
          f"{'Old Mem MB':>10} {'New Mem MB':>10} {'ΔMem MB':>10}")
    for model_name, rows in by_model.items():
        rmses = [float(r["rmse"]) for r in rows]
        mems  = [float(r["memory_mb"]) for r in rows]
        prev  = PREVIOUS_NESTED_DICT.get(model_name, {})
        if prev:
            print(
                f"{model_name:<30} {prev['mean_rmse']:>10.4f} {_mean(rmses):>10.4f} "
                f"{_mean(rmses) - prev['mean_rmse']:>+8.4f} "
                f"{prev['mean_memory_mb']:>10.2f} {_mean(mems):>10.2f} "
                f"{_mean(mems) - prev['mean_memory_mb']:>+10.2f}"
            )
    print(f"\nSaved: {folds_path.name}, {summary_path.name}")


if __name__ == "__main__":
    main()

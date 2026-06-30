from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
REPORT_IMAGE_DIR = PROJECT_DIR.parent / "推荐系统实验报告" / "images"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mf_factor_count(row: dict[str, str]) -> int:
    if row.get("n_factors"):
        return int(row["n_factors"])
    match = re.search(r"_f(\d+)_", row.get("name", ""))
    if not match:
        raise ValueError(f"Cannot infer n_factors from row: {row}")
    return int(match.group(1))


def save_model_comparison() -> None:
    rows = read_csv(RESULTS_DIR / "experiment_results.csv")
    names = [row["model"].replace("MatrixFactorization", "MF") for row in rows]
    mean_rmse = [float(row["mean_rmse"]) for row in rows]
    std_rmse = [float(row["std_rmse"]) for row in rows]
    train_time = [float(row["avg_train_seconds"]) for row in rows]
    memory = [float(row["avg_memory_mb"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    colors = ["#7aa6c2", "#8dbf75", "#d6a45f", "#c96f6f"]

    axes[0].bar(names, mean_rmse, yerr=std_rmse, capsize=4, color=colors)
    axes[0].set_title("Prediction Accuracy")
    axes[0].set_ylabel("5-fold mean RMSE")
    axes[0].set_ylim(min(mean_rmse) - 0.08, max(mean_rmse) + 0.08)
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axes[1].bar(names, train_time, color=colors)
    axes[1].set_title("Training Time")
    axes[1].set_ylabel("Average train time (s)")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axes[2].bar(names, memory, color=colors)
    axes[2].set_title("Model Storage")
    axes[2].set_ylabel("Approximate memory (MB)")
    axes[2].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    for ax in axes:
        ax.tick_params(axis="x", rotation=18)

    fig.savefig(REPORT_IMAGE_DIR / "model_main_comparison.png", dpi=300)
    plt.close(fig)


def save_cv_fold_comparison() -> None:
    rows = read_csv(RESULTS_DIR / "cv_fold_count_results.csv")
    folds = [int(row["n_folds"]) for row in rows]
    mean_rmse = [float(row["mean_rmse"]) for row in rows]
    std_rmse = [float(row["std_rmse"]) for row in rows]
    total_time = [float(row["total_train_seconds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(folds, mean_rmse, marker="o", linewidth=2.2, color="#1f77b4")
    ax.errorbar(folds, mean_rmse, yerr=std_rmse, fmt="none", capsize=4, color="#1f77b4")
    ax.set_title("Cross-Validation Fold Count Trade-off")
    ax.set_xlabel("number of folds")
    ax.set_ylabel("Mean RMSE", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_xticks(folds)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.axvline(5, color="#777777", linestyle=":", linewidth=1.6)

    ax2 = ax.twinx()
    ax2.plot(folds, total_time, marker="s", linestyle="--", linewidth=2.0, color="#d95f02")
    ax2.set_ylabel("Total train time (s)", color="#d95f02")
    ax2.tick_params(axis="y", labelcolor="#d95f02")

    fig.savefig(REPORT_IMAGE_DIR / "cv_fold_count_tradeoff.png", dpi=300)
    plt.close(fig)


def save_mf_tuning() -> None:
    high_rows = read_csv(RESULTS_DIR / "mf_parameter_results.csv")
    residual_rows = read_csv(RESULTS_DIR / "mf_residual_cv_results.csv")
    low_rows = read_csv(RESULTS_DIR / "mf_residual_narrow_cv_results.csv")
    selected_rows = read_csv(RESULTS_DIR / "exploration_cv_selected.csv")

    high_rows = sorted(high_rows, key=lambda row: int(row["n_factors"]))
    rows_by_name = {
        row["name"]: row
        for row in residual_rows + low_rows + selected_rows
        if "name" in row
    }
    residual_candidates = [
        rows_by_name["mf_residual_1d_lr003_reg01"],
        rows_by_name["current_2d_ep12_lr002_reg02"],
        rows_by_name["3d_ep12_lr002_reg02"],
        rows_by_name["4d_ep12_lr002_reg02"],
        rows_by_name["MF_f5_lr002_reg03"],
        rows_by_name["MF_f8_lr002_reg03"],
    ]
    residual_candidates = sorted(residual_candidates, key=mf_factor_count)
    low_2d_variants = [
        row
        for row in low_rows
        if row["name"]
        in {
            "current_2d_ep12_lr002_reg02",
            "2d_ep12_lr0015_reg02",
            "2d_ep12_lr0025_reg02",
            "2d_ep12_lr002_reg01",
            "2d_ep12_lr002_reg03",
            "2d_ep16_lr002_reg02",
        }
    ]
    low_2d_variants = sorted(low_2d_variants, key=lambda row: float(row["mean_rmse"]))

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    ax.plot(
        [int(row["n_factors"]) for row in high_rows],
        [float(row["rmse"]) for row in high_rows],
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
    )
    ax.set_title("Original MF High-Dimension Instability")
    ax.set_xlabel("n_factors")
    ax.set_ylabel("Validation RMSE")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    fig.savefig(REPORT_IMAGE_DIR / "mf_initial_dimension_scan.png", dpi=300)
    plt.close(fig)

    dims = [mf_factor_count(row) for row in residual_candidates]
    mean_rmse = [float(row["mean_rmse"]) for row in residual_candidates]
    std_rmse = [float(row["std_rmse"]) for row in residual_candidates]
    memory_mb = [float(row["avg_memory_mb"]) for row in residual_candidates]
    labels = [
        "1D\nbest",
        "2D",
        "3D",
        "4D",
        "5D\nreg=0.3",
        "8D\nfinal",
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.3), constrained_layout=True)
    ax.errorbar(
        dims,
        mean_rmse,
        yerr=std_rmse,
        marker="o",
        markersize=7,
        capsize=4,
        linewidth=2.3,
        color="#2f7f4f",
        label="5-fold mean RMSE",
    )
    ax.scatter([dims[-1]], [mean_rmse[-1]], s=95, color="#c96f6f", zorder=5, label="final MF")
    ax.set_title("Residual MF Accuracy-Storage Trade-off")
    ax.set_xlabel("n_factors")
    ax.set_ylabel("5-fold mean RMSE")
    ax.set_xticks(dims)
    ax.set_xticklabels(labels)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_ylim(min(mean_rmse) - 0.055, max(mean_rmse) + 0.055)

    ax2 = ax.twinx()
    ax2.plot(
        dims,
        memory_mb,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        color="#7aa6c2",
        label="core storage",
    )
    ax2.set_ylabel("Core storage (MB)", color="#527d96")
    ax2.tick_params(axis="y", labelcolor="#527d96")

    for x_value, rmse_value in zip(dims, mean_rmse):
        ax.text(x_value, rmse_value - 0.018, f"{rmse_value:.3f}", ha="center", va="top", fontsize=8.5)

    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, line_labels + line_labels2, loc="upper right", fontsize=8.5)
    fig.savefig(REPORT_IMAGE_DIR / "mf_residual_dimension_scan.png", dpi=300)
    plt.close(fig)

    labels = [
        row["name"]
        .replace("current_", "")
        .replace("_ep", "\nep")
        .replace("_lr", " lr")
        .replace("_reg", " reg")
        for row in low_2d_variants
    ]
    best_rmse = min(float(row["mean_rmse"]) for row in low_2d_variants)
    deltas = [float(row["mean_rmse"]) - best_rmse for row in low_2d_variants]
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    bars = ax.barh(labels, deltas, color="#d6a45f")
    ax.set_title("2D Residual MF Local Search")
    ax.set_xlabel("RMSE increase vs best")
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_xlim(0, max(deltas) * 1.18 if max(deltas) > 0 else 0.01)
    for bar, row, delta in zip(bars, low_2d_variants, deltas):
        ax.text(
            bar.get_width() + max(deltas) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"+{delta:.4f}  ({float(row['mean_rmse']):.4f})",
            va="center",
            fontsize=8.5,
        )
    fig.savefig(REPORT_IMAGE_DIR / "mf_local_search.png", dpi=300)
    plt.close(fig)


def save_ensemble_weight_curve() -> None:
    rows = read_csv(RESULTS_DIR / "ensemble_weight_fine_results.csv")
    rows = sorted(rows, key=lambda row: float(row["itemcf_weight"]))
    weights = [float(row["itemcf_weight"]) for row in rows]
    mean_rmse = [float(row["mean_rmse"]) for row in rows]
    std_rmse = [float(row["std_rmse"]) for row in rows]

    best_index = min(range(len(rows)), key=lambda idx: mean_rmse[idx])
    best_weight = weights[best_index]
    best_rmse = mean_rmse[best_index]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), constrained_layout=True)

    axes[0].plot(weights, mean_rmse, marker="o", markersize=4, linewidth=2.0, color="#1f77b4")
    axes[0].fill_between(
        weights,
        [m - s for m, s in zip(mean_rmse, std_rmse)],
        [m + s for m, s in zip(mean_rmse, std_rmse)],
        color="#1f77b4",
        alpha=0.12,
    )
    axes[0].axvline(best_weight, color="#777777", linestyle=":", linewidth=1.6)
    axes[0].set_title("Full Weight Sweep")
    axes[0].set_xlabel("ItemCF weight alpha")
    axes[0].set_ylabel("5-fold mean RMSE")
    axes[0].grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    close_rows = [row for row in rows if 0.45 <= float(row["itemcf_weight"]) <= 0.65]
    close_weights = [float(row["itemcf_weight"]) for row in close_rows]
    close_rmse = [float(row["mean_rmse"]) for row in close_rows]
    axes[1].plot(close_weights, close_rmse, marker="o", markersize=4, linewidth=2.0, color="#2ca02c")
    axes[1].axvline(best_weight, color="#777777", linestyle=":", linewidth=1.6)
    axes[1].scatter([best_weight], [best_rmse], color="#d62728", zorder=3)
    axes[1].set_title("Fine Search Near the Optimum")
    axes[1].set_xlabel("ItemCF weight alpha")
    axes[1].set_ylabel("5-fold mean RMSE")
    axes[1].grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    fig.savefig(REPORT_IMAGE_DIR / "ensemble_weight_curve.png", dpi=300)
    plt.close(fig)


def save_optimized_model_comparison() -> None:
    old_rows = read_csv(RESULTS_DIR / "exploration_best_methods_multiseed_summary.csv")
    compact_rows = read_csv(RESULTS_DIR / "compact_storage_comparison.csv")
    selected = ["Ensemble_official", "ItemCF_tuned_baseline", "MF_tuned", "UserCF_tuned", "Ensemble_triple_tuned"]
    row_by_model = {row["model"]: row for row in old_rows}
    row_by_model.update({row["model"]: row for row in compact_rows})
    labels = [
        "Old\nEnsemble",
        "Tuned\nItemCF",
        "Tuned\nMF",
        "UserCF",
        "Tuned\nTriple",
    ]
    selected_rows = [row_by_model[name] for name in selected]
    mean_rmse = [float(row["mean_rmse"]) for row in selected_rows]
    std_rmse = [float(row["std_rmse"]) for row in selected_rows]
    train_key = "mean_train_s" if "mean_train_s" in selected_rows[-1] else "mean_train_seconds"
    memory_key = "mean_memory_mb"
    train_time = [
        float(row.get(train_key) or row.get("mean_train_seconds"))
        for row in selected_rows
    ]
    memory = [float(row[memory_key]) for row in selected_rows]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)
    colors = ["#7aa6c2", "#8dbf75", "#d6a45f", "#b894c6", "#5f9e8f"]

    axes[0].bar(labels, mean_rmse, yerr=std_rmse, capsize=4, color=colors)
    axes[0].set_title("Multi-seed Accuracy")
    axes[0].set_ylabel("25-fold mean RMSE")
    axes[0].set_ylim(min(mean_rmse) - 0.06, max(mean_rmse) + 0.08)
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axes[1].bar(labels, train_time, color=colors)
    axes[1].set_title("Training Time")
    axes[1].set_ylabel("Mean train time (s)")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    axes[2].bar(labels, memory, color=colors)
    axes[2].set_title("Model Storage")
    axes[2].set_ylabel("Approximate memory (MB)")
    axes[2].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    fig.savefig(REPORT_IMAGE_DIR / "optimized_model_comparison.png", dpi=300)
    plt.close(fig)


def save_multiseed_fold_improvement() -> None:
    old_rows = read_csv(RESULTS_DIR / "exploration_best_methods_multiseed_folds.csv")
    new_rows = read_csv(RESULTS_DIR / "compact_storage_comparison_folds.csv")
    by_fold: dict[tuple[str, str], dict[str, float]] = {}
    for row in old_rows:
        key = (row["seed"], row["fold"])
        by_fold.setdefault(key, {})[row["model"]] = float(row["rmse"])
    for row in new_rows:
        key = (row["seed"], row["fold"])
        by_fold.setdefault(key, {})[row["model"]] = float(row["rmse"])

    ordered_keys = sorted(by_fold, key=lambda key: (int(key[0]), int(key[1])))
    x_values = list(range(1, len(ordered_keys) + 1))
    old_rmse = [by_fold[key]["Ensemble_official"] for key in ordered_keys]
    new_rmse = [by_fold[key]["Ensemble_triple_tuned"] for key in ordered_keys]
    improvements = [old - new for old, new in zip(old_rmse, new_rmse)]

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.0), constrained_layout=True)

    axes[0].plot(x_values, old_rmse, marker="o", linewidth=1.8, label="Old Ensemble", color="#7aa6c2")
    axes[0].plot(x_values, new_rmse, marker="s", linewidth=1.8, label="Tuned Triple Ensemble", color="#c96f6f")
    axes[0].set_title("RMSE Across 25 Validation Folds")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    axes[0].legend()

    axes[1].bar(x_values, improvements, color="#5f9e8f")
    axes[1].axhline(0, color="#555555", linewidth=1.0)
    axes[1].set_title("Paired Improvement on Each Fold")
    axes[1].set_xlabel("validation fold index across five random seeds")
    axes[1].set_ylabel("Old RMSE - New RMSE")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    fig.savefig(REPORT_IMAGE_DIR / "multiseed_fold_improvement.png", dpi=300)
    plt.close(fig)


def main() -> None:
    REPORT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    save_cv_fold_comparison()
    save_model_comparison()
    save_mf_tuning()
    save_ensemble_weight_curve()
    save_optimized_model_comparison()
    save_multiseed_fold_improvement()


if __name__ == "__main__":
    main()

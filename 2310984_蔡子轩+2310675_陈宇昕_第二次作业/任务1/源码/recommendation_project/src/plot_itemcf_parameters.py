from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_DIR / "results" / "itemcf_parameter_results.csv"
REPORT_IMAGE_DIR = PROJECT_DIR.parent / "推荐系统实验报告" / "images"

BASELINE = {
    "shrinkage": 10.0,
    "min_common": 2.0,
    "top_k": 20.0,
    "similarity_top_n": 80.0,
    "block_size": 256.0,
}


def load_rows() -> list[dict[str, str]]:
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_for(rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    group_rows = [row for row in rows if row["sweep_group"] == group]
    return sorted(group_rows, key=lambda row: float(row["parameter_value"]))


def plot_rmse_only(
    ax: plt.Axes,
    group_rows: list[dict[str, str]],
    title: str,
    xlabel: str,
    baseline_value: float | None = None,
) -> None:
    x = [float(row["parameter_value"]) for row in group_rows]
    rmse = [float(row["rmse"]) for row in group_rows]

    ax.plot(x, rmse, marker="o", linewidth=2.2, color="#2563a8")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation RMSE")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

    if baseline_value is not None:
        ax.axvline(baseline_value, color="#777777", linestyle=":", linewidth=1.6)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))


def plot_rmse_with_memory(
    ax: plt.Axes,
    group_rows: list[dict[str, str]],
    title: str,
    xlabel: str,
    baseline_value: float | None = None,
) -> None:
    x = [float(row["parameter_value"]) for row in group_rows]
    rmse = [float(row["rmse"]) for row in group_rows]
    memory = [float(row["memory_mb"]) for row in group_rows]

    ax.plot(x, rmse, marker="o", linewidth=2.2, color="#2563a8", label="RMSE")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation RMSE", color="#2563a8")
    ax.tick_params(axis="y", labelcolor="#2563a8")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    if baseline_value is not None:
        ax.axvline(baseline_value, color="#777777", linestyle=":", linewidth=1.6)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        memory,
        marker="s",
        linewidth=1.9,
        linestyle="--",
        color="#c96f36",
    )
    ax2.set_ylabel("Memory (MB)", color="#c96f36")
    ax2.tick_params(axis="y", labelcolor="#c96f36")
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))


def save_similarity_reliability(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.8), constrained_layout=True)
    plot_rmse_only(
        axes[0],
        rows_for(rows, "shrinkage"),
        "Effect of shrinkage on RMSE",
        "shrinkage",
        BASELINE["shrinkage"],
    )
    plot_rmse_only(
        axes[1],
        rows_for(rows, "min_common"),
        "Effect of min_common on RMSE",
        "min_common",
        BASELINE["min_common"],
    )
    fig.savefig(REPORT_IMAGE_DIR / "itemcf_similarity_reliability.png", dpi=300)
    plt.close(fig)


def save_neighbor_scale(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.8), constrained_layout=True)
    plot_rmse_only(
        axes[0],
        rows_for(rows, "top_k"),
        "Prediction neighbors reach a plateau",
        "top_k",
        BASELINE["top_k"],
    )
    plot_rmse_with_memory(
        axes[1],
        rows_for(rows, "similarity_top_n"),
        "Stored neighbors: RMSE vs. memory",
        "similarity_top_n",
        BASELINE["similarity_top_n"],
    )
    fig.savefig(REPORT_IMAGE_DIR / "itemcf_neighbor_scale.png", dpi=300)
    plt.close(fig)


def save_block_size_time(rows: list[dict[str, str]]) -> None:
    block_rows = rows_for(rows, "block_size")
    x = [float(row["parameter_value"]) for row in block_rows]
    train_seconds = [float(row["train_seconds"]) for row in block_rows]
    rmse = [float(row["rmse"]) for row in block_rows]

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    ax.plot(
        x,
        train_seconds,
        marker="o",
        linewidth=2.2,
        color="#2563a8",
    )
    ax.set_title("Block size mainly changes training time", fontsize=13)
    ax.set_xlabel("block_size")
    ax.set_ylabel("Train time (s)")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.axvline(BASELINE["block_size"], color="#777777", linestyle=":", linewidth=1.6)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    rmse_text = f"RMSE nearly unchanged: {min(rmse):.6f}-{max(rmse):.6f}"
    ax.text(
        0.98,
        0.92,
        rmse_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.savefig(REPORT_IMAGE_DIR / "itemcf_block_size_time.png", dpi=300)
    plt.close(fig)


def main() -> None:
    REPORT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    save_similarity_reliability(rows)
    save_neighbor_scale(rows)
    save_block_size_time(rows)


if __name__ == "__main__":
    main()

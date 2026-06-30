"""Train one model on full Train.txt and generate final_result.txt."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline import BaselineModel
from ensemble import EnsembleModel
from item_cf import ItemCFModel
from matrix_factorization import MatrixFactorizationModel
from optimized_ensemble import OptimizedEnsembleModel
from utils import clip_rating, read_test, read_train, write_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENT_RESULTS_PATH = RESULTS_DIR / "experiment_results.csv"


def build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "baseline":
        return BaselineModel(seed=2026)
    if model_name == "itemcf":
        return ItemCFModel(
            top_k=40,
            similarity_top_n=200,
            shrinkage=50.0,
            min_common=2,
            block_size=256,
            seed=2026,
        )
    if model_name in {"mf", "matrix_factorization"}:
        return MatrixFactorizationModel(
            n_factors=4,
            n_epochs=12,
            learning_rate=0.002,
            reg=0.2,
            reg_bias=0.05,
            init_std=0.01,
            init_with_baseline=True,
            freeze_bias=True,
            seed=2026,
        )
    if model_name == "ensemble":
        return EnsembleModel(itemcf_weight=0.55, seed=2026)
    if model_name in {"optimized_ensemble", "optimized"}:
        return OptimizedEnsembleModel(seed=2026)
    raise ValueError(f"Unsupported model: {model_name}")


def select_best_model(results_path: Path) -> str:
    """Read experiment_results.csv and return the lowest-RMSE model."""

    if not results_path.exists():
        raise FileNotFoundError(
            "experiment_results.csv not found. Run `python3 src/run_experiment.py` "
            "first, or pass --model explicitly."
        )

    csv_to_cli_name = {
        "Baseline": "baseline",
        "ItemCF": "itemcf",
        "MatrixFactorization": "mf",
        "Ensemble": "ensemble",
    }

    with open(results_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("experiment_results.csv is empty.")

    rmse_key = "mean_rmse" if "mean_rmse" in rows[0] else "rmse"
    best_row = min(rows, key=lambda row: float(row[rmse_key]))
    model_name = best_row["model"]
    if model_name not in csv_to_cli_name:
        raise ValueError(f"Unsupported model name in CSV: {model_name}")
    return csv_to_cli_name[model_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final rating predictions.")
    parser.add_argument(
        "--model",
        default="auto",
        choices=[
            "auto",
            "baseline",
            "itemcf",
            "mf",
            "matrix_factorization",
            "ensemble",
            "optimized_ensemble",
            "optimized",
        ],
        help="Model used for final prediction. `auto` selects the best 5-fold CV model.",
    )
    args = parser.parse_args()

    train_path = DATA_DIR / "Train.txt"
    test_path = DATA_DIR / "Test.txt"
    result_form_path = DATA_DIR / "ResultForm.txt"
    output_path = RESULTS_DIR / "final_result.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_data = read_train(str(train_path))
    test_pairs = read_test(str(test_path))

    selected_model_name = args.model
    if selected_model_name == "auto":
        selected_model_name = select_best_model(EXPERIMENT_RESULTS_PATH)

    model = build_model(selected_model_name)
    model.fit(train_data)

    rating_min = min(rating for _, _, rating in train_data)
    rating_max = max(rating for _, _, rating in train_data)
    predictions = model.batch_predict(test_pairs)
    clipped_predictions = [
        (user_id, item_id, clip_rating(score, rating_min, rating_max))
        for user_id, item_id, score in predictions
    ]

    write_predictions(
        str(output_path),
        clipped_predictions,
        result_form_path=str(result_form_path),
    )
    if args.model == "auto":
        print(
            "Model: auto "
            f"(selected by lowest validation RMSE from {EXPERIMENT_RESULTS_PATH.name})"
        )
    else:
        print(f"Model: {args.model}")
    print(f"Selected model: {selected_model_name}")
    print(f"Predictions: {len(clipped_predictions)}")
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()

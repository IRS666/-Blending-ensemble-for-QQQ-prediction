"""CLI entry point for LR/SVC/KNN -> XGBoost blending training."""

import argparse
import json

import joblib
import pandas as pd

from feature_selection import SelectionConfig
from model_pipeline import ModelConfig, train_blending_ensemble
from nvconfig import MODELS_DIR, PROCESSED_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-auto-tune", action="store_true")
    parser.add_argument("--search-iterations", type=int, default=12)
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--optimize-threshold", action="store_true")
    parser.add_argument("--threshold-objective", choices=("total_return", "sharpe"), default="total_return")
    parser.add_argument(
        "--optimized-preset",
        action="store_true",
        help="Use the locked development hyperparameters validated on Holdout.",
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--evaluation-mode", choices=("static", "walk_forward"), default="static")
    parser.add_argument("--walk-forward-frequency", choices=("monthly", "quarterly"), default="monthly")
    args = parser.parse_args()

    dataset_path = PROCESSED_DIR / "selected_dataset.csv"
    transformer_path = MODELS_DIR / "feature_transformer.joblib"
    if not dataset_path.exists() or not transformer_path.exists():
        raise FileNotFoundError("Run run_feature_engineering.py before model training.")
    dataset = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    transformer = joblib.load(transformer_path)
    config = ModelConfig(
        auto_tune=not args.no_auto_tune,
        search_iterations=args.search_iterations,
        search_metric=args.metric,
        prediction_threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        threshold_objective=args.threshold_objective,
        transaction_cost_bps=args.cost_bps,
        evaluation_mode=args.evaluation_mode,
        walk_forward_frequency=args.walk_forward_frequency,
    )
    if args.optimized_preset:
        config.lr_grid = {
            "model__C": [0.1], "model__penalty": ["l1"],
            "model__fit_intercept": [True], "model__tol": [1e-4],
            "model__class_weight": [None],
        }
        config.svc_grid = {
            "model__C": [0.3], "model__gamma": [0.00003],
            "model__kernel": ["rbf"], "model__shrinking": [True],
            "model__tol": [1e-4], "model__class_weight": ["balanced"],
            "model__degree": [3], "model__coef0": [0.0],
        }
        config.knn_grid = {
            "model__n_neighbors": [21], "model__weights": ["distance"],
            "model__p": [2], "model__leaf_size": [20],
            "model__algorithm": ["auto"],
        }
        config.meta_grid = {
            "n_estimators": [60], "max_depth": [1], "learning_rate": [0.08],
            "subsample": [1.0], "colsample_bytree": [1.0],
            "min_child_weight": [10.0], "reg_alpha": [0.0],
            "reg_lambda": [1.0], "gamma": [0.1], "grow_policy": ["depthwise"],
            "max_leaves": [0], "max_bin": [128],
        }
    summary = train_blending_ensemble(dataset, transformer, config)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

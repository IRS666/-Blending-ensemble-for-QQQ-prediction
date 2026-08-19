"""Search development configurations without evaluating Final Validation."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_pipeline import load_market_and_macro_data
from feature_pipeline import create_feature_universe
from feature_selection import SelectionConfig, run_feature_selection
from model_pipeline import ModelConfig, optimize_blending_development
from nvconfig import MODELS_DIR, OUTPUT_DIR, PROCESSED_DIR
from split_utils import chronological_blending_split
from wavelet_pipeline import WaveletConfig, denoise_market_data


SEARCH_DIR = OUTPUT_DIR / "development_search"


def _parse_numbers(raw, cast=float):
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-02")
    parser.add_argument("--source", choices=("raw", "wavelet"), default="raw")
    parser.add_argument("--target-thresholds", default="0.005,0.0075,0.01,0.0125,0.015")
    parser.add_argument("--feature-counts", default="20,30,40,50,60")
    parser.add_argument("--base-iterations", type=int, default=100)
    parser.add_argument("--meta-iterations", type=int, default=60)
    parser.add_argument("--expanded-grid", action="store_true")
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=2)
    parser.add_argument("--wavelet-window", type=int, default=128)
    args = parser.parse_args()

    thresholds = _parse_numbers(args.target_thresholds)
    feature_counts = _parse_numbers(args.feature_counts, int)
    raw, provenance = load_market_and_macro_data(
        start=args.start, include_credit_spreads=True, refresh=False
    )
    source = raw
    wavelet_summary = {"enabled": False}
    if args.source == "wavelet":
        source, wavelet_summary, _ = denoise_market_data(
            raw,
            WaveletConfig(
                wavelet=args.wavelet,
                level=args.wavelet_level,
                threshold_rule="universal_mad",
                threshold_mode="soft",
                window_size=args.wavelet_window,
                minimum_history=32,
                threshold_scale=1.0,
                field_mode="prices_and_volume",
                n_jobs=-1,
            ),
        )

    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for target_threshold in thresholds:
        scenario_id = f"{args.source}_target-{target_threshold:.4f}".replace(".", "p")
        scenario_dir = SEARCH_DIR / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        universe = create_feature_universe(
            source,
            target_threshold=target_threshold,
            target_data=raw,
            use_cache=True,
            persist_outputs=True,
        )
        split = chronological_blending_split(universe)
        selection_config = SelectionConfig(
            missing_threshold=0.20,
            minimum_abs_correlation=0.015,
            minimum_mi_quantile=0.50,
            stage1_max_features=400,
            stage2_metric="roc_auc",
            stage2_min_metric=0.50,
            stage2_top_features=160,
            multicollinearity_threshold=0.92,
            embedded_method="extra_trees",
            embedded_top_features=max(feature_counts),
            rfe_enabled=False,
            final_feature_count=max(feature_counts),
            cv_splits=4,
            n_jobs=-1,
        )
        _, _, feature_summary = run_feature_selection(
            universe, split["subset_train"].index, selection_config
        )
        state = joblib.load(PROCESSED_DIR / "feature_selection_state.joblib")
        ranked_features = state["embedded_ranking"].sort_values(
            "embedded_importance", ascending=False
        )["feature"].tolist()
        model_config = ModelConfig(
            auto_tune=True,
            base_search_iterations=args.base_iterations,
            meta_search_iterations=args.meta_iterations,
            base_cv_splits=4,
            meta_cv_splits=5,
            search_metric="roc_auc",
            optimize_threshold=True,
            threshold_objective="total_return",
            transaction_cost_bps=10.0,
            meta_class_weight="balanced",
            meta_monotone_probabilities=True,
            evaluation_mode="static",
        )
        if args.expanded_grid:
            model_config.lr_grid = {
                "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__penalty": ["l1", "l2"],
                "model__fit_intercept": [True],
                "model__tol": [1e-5, 1e-4],
                "model__class_weight": [None, "balanced"],
            }
            model_config.svc_grid = {
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "model__gamma": [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, "scale"],
                "model__kernel": ["rbf"],
                "model__shrinking": [True, False],
                "model__tol": [1e-4, 1e-3, 1e-2],
                "model__class_weight": [None, "balanced"],
                "model__degree": [3],
                "model__coef0": [0.0],
            }
            model_config.knn_grid = {
                "model__n_neighbors": [11, 15, 21, 31, 41, 61, 81, 101, 121],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
                "model__leaf_size": [20, 30, 50],
                "model__algorithm": ["auto"],
            }
            model_config.meta_grid = {
                "n_estimators": [20, 40, 60, 80, 120, 160],
                "max_depth": [1, 2, 3],
                "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.67, 1.0],
                "min_child_weight": [2.0, 5.0, 10.0, 20.0],
                "reg_alpha": [0.0, 0.05, 0.1, 0.5, 1.0],
                "reg_lambda": [1.0, 3.0, 5.0, 10.0, 20.0],
                "gamma": [0.0, 0.05, 0.1, 0.2],
                "grow_policy": ["depthwise"],
                "max_leaves": [0],
                "max_bin": [128, 256],
            }
        winner_frame, winner_transformer, selection, ranking = optimize_blending_development(
            universe,
            state["eda"],
            ranked_features,
            model_config,
            feature_counts,
        )
        winner = selection["winner"]
        row = {
            "scenario_id": scenario_id,
            "source": args.source,
            "target_threshold": float(target_threshold),
            "feature_count": int(winner["feature_count"]),
            "holdout_oof_roc_auc": float(winner["meta_cv_score"]),
            "holdout_oof_strategy_return": float(winner["development_strategy_return"]),
            "holdout_oof_strategy_sharpe": float(winner["development_strategy_sharpe"]),
            "holdout_oof_strategy_exposure": float(winner["development_strategy_exposure"]),
            "development_threshold": float(winner["development_threshold"]),
            "mean_base_cv_roc_auc": float(winner["base_cv_score_mean"]),
            "final_validation_metrics_read": False,
        }
        all_rows.append(row)
        winner_frame.to_csv(scenario_dir / "selected_dataset.csv")
        joblib.dump(winner_transformer, scenario_dir / "feature_transformer.joblib")
        ranking.to_csv(scenario_dir / "feature_count_candidates.csv", index=False)
        (scenario_dir / "scenario.json").write_text(
            json.dumps(
                _json_ready(
                    {
                        **row,
                        "features": winner["feature_names"],
                        "feature_selection": feature_summary,
                        "candidate_selection": selection,
                        "data_provenance": provenance,
                        "wavelet": wavelet_summary,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    current = pd.DataFrame(all_rows)
    aggregate_path = SEARCH_DIR / "scenario_results.csv"
    if aggregate_path.exists():
        previous = pd.read_csv(aggregate_path)
        current = pd.concat([previous, current], ignore_index=True)
        current = current.drop_duplicates("scenario_id", keep="last")
    current["roc_rank"] = current["holdout_oof_roc_auc"].rank(
        ascending=False, method="min"
    )
    current["return_rank"] = current["holdout_oof_strategy_return"].rank(
        ascending=False, method="min"
    )
    current["sharpe_rank"] = current["holdout_oof_strategy_sharpe"].rank(
        ascending=False, method="min"
    )
    current["joint_rank_score"] = (
        current["roc_rank"] + current["return_rank"] + 0.5 * current["sharpe_rank"]
    )
    current = current.sort_values(
        ["joint_rank_score", "holdout_oof_roc_auc", "holdout_oof_strategy_return"],
        ascending=[True, False, False],
    )
    current.to_csv(aggregate_path, index=False)
    print("DEVELOPMENT-ONLY RANKING")
    print(current.to_string(index=False))
    print(f"Outputs: {SEARCH_DIR}")


if __name__ == "__main__":
    main()

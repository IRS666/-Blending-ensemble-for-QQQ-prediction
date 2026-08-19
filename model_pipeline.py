"""Leakage-controlled LR/SVC/KNN base learners and XGBoost meta learner."""

from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from nvconfig import (
    FIGURES_DIR, MODELS_DIR, OUTPUT_DIR, PROCESSED_DIR, ROOT, SEED, TABLES_DIR,
    ensure_directories,
)
from split_utils import chronological_blending_split


@dataclass
class ModelConfig:
    auto_tune: bool = True
    search_iterations: int = 50
    base_search_iterations: int | None = 100
    meta_search_iterations: int | None = 60
    base_cv_splits: int = 4
    meta_cv_splits: int = 5
    search_metric: str = "roc_auc"
    prediction_threshold: float = 0.50
    optimize_threshold: bool = False
    threshold_objective: str = "total_return"
    threshold_grid_size: int = 101
    threshold_min_exposure: float = 0.05
    transaction_cost_bps: float = 10.0
    meta_class_weight: str = "balanced"
    meta_monotone_probabilities: bool = True
    evaluation_mode: str = "static"
    walk_forward_frequency: str = "monthly"
    auto_optimization_details: dict | None = None
    lr_grid: dict = field(default_factory=lambda: {
        "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
        "model__penalty": ["l1", "l2"], "model__fit_intercept": [True],
        "model__tol": [1e-4], "model__class_weight": [None, "balanced"],
    })
    svc_grid: dict = field(default_factory=lambda: {
        "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "model__gamma": [0.0001, 0.0003, 0.001, 0.003, "scale"],
        "model__kernel": ["rbf"], "model__shrinking": [True],
        "model__tol": [1e-3, 1e-2], "model__class_weight": [None, "balanced"],
        "model__degree": [3], "model__coef0": [0.0],
    })
    knn_grid: dict = field(default_factory=lambda: {
        "model__n_neighbors": [21, 31, 41, 51, 81],
        "model__weights": ["uniform", "distance"], "model__p": [1, 2],
        "model__leaf_size": [30], "model__algorithm": ["auto"],
    })
    meta_grid: dict = field(default_factory=lambda: {
        "n_estimators": [20, 30, 50, 80], "max_depth": [1, 2],
        "learning_rate": [0.01, 0.03, 0.05], "subsample": [0.85, 1.0],
        "colsample_bytree": [1.0], "min_child_weight": [5.0, 10.0, 20.0],
        "reg_alpha": [0.0, 0.1, 0.5], "reg_lambda": [1.0, 5.0, 10.0],
        "gamma": [0.0, 0.05, 0.1], "grow_policy": ["depthwise"],
        "max_leaves": [0], "max_bin": [256],
    })


def _scorer_name(metric):
    return "f1" if metric == "f1" else metric


def _search(estimator, grid, X, y, config, name, cv_splits, results_path=None):
    """Tune one estimator with chronological CV.

    ``results_path`` keeps development-only research artifacts separate from
    the tables of the eventual, single Final Validation run.
    """
    total = len(list(ParameterGrid(grid)))
    configured_budget = config.meta_search_iterations if name == "xgboost_meta" else config.base_search_iterations
    search_budget = config.search_iterations if configured_budget is None else configured_budget
    cv = TimeSeriesSplit(n_splits=cv_splits, gap=1)
    if not config.auto_tune:
        params = {key: list(values)[0] for key, values in grid.items()}
        fitted = clone(estimator).set_params(**params).fit(X, y)
        return fitted, {"best_params": params, "best_score": None, "strategy": "fixed", "candidates": 1}
    if total <= search_budget:
        search = GridSearchCV(estimator, grid, scoring=_scorer_name(config.search_metric), cv=cv, n_jobs=-1, refit=True)
        strategy = "grid"
    else:
        search = RandomizedSearchCV(
            estimator, grid, n_iter=min(search_budget, total),
            scoring=_scorer_name(config.search_metric), cv=cv, n_jobs=-1,
            refit=True, random_state=SEED,
        )
        strategy = "randomized"
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results.to_csv(
        results_path or TABLES_DIR / f"tuning_{name}.csv", index=False
    )
    return search.best_estimator_, {
        "best_params": search.best_params_, "best_score": float(search.best_score_),
        "strategy": strategy, "candidates": int(len(results)), "search_budget": int(search_budget),
    }


def _base_estimators(transformer):
    return {
        "lr": Pipeline([
            ("transform", clone(transformer)),
            ("model", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000, random_state=SEED)),
        ]),
        "svc": Pipeline([
            ("transform", clone(transformer)),
            ("model", SVC(probability=True, class_weight="balanced", random_state=SEED)),
        ]),
        "knn": Pipeline([
            ("transform", clone(transformer)),
            ("model", KNeighborsClassifier()),
        ]),
    }


def _xgb_estimator(config):
    monotone_constraints = (1, 1, 1) if config.meta_monotone_probabilities else None
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_jobs=-1, random_state=SEED, monotone_constraints=monotone_constraints,
    )


def _probability_frame(models, X):
    return pd.DataFrame(
        {f"{name}_probability": model.predict_proba(X)[:, 1] for name, model in models.items()},
        index=X.index,
    )


def _walk_forward_period_blocks(frame, frequency):
    """Return chronological monthly/quarterly prediction blocks."""

    period_code = {"monthly": "M", "quarterly": "Q"}.get(frequency)
    if period_code is None:
        raise ValueError("walk_forward_frequency must be 'monthly' or 'quarterly'.")
    return [block.copy() for _, block in frame.groupby(frame.index.to_period(period_code), sort=True)]


def _fit_frozen_blend_window(history, prediction_block, transformer, features, tuning, config):
    """Refit one strict holdout blend using frozen hyperparameters and past data only."""

    development_split = chronological_blending_split(
        history,
        subset_ratio=0.75,
        holdout_ratio=0.25,
        validation_ratio=0.0,
        purge_size=1,
    )
    subset = development_split["subset_train"]
    holdout = development_split["holdout"]
    if min(len(subset), len(holdout)) < 20:
        raise ValueError("Insufficient history for a walk-forward blending refit.")

    base_models = {}
    for name, estimator in _base_estimators(transformer).items():
        frozen_params = tuning[name]["best_params"]
        base_models[name] = clone(estimator).set_params(**frozen_params)
        base_models[name].fit(subset[features], subset["target"].astype(int))

    subset_meta = _probability_frame(base_models, subset[features])
    holdout_meta = _probability_frame(base_models, holdout[features])
    y_holdout = holdout["target"].astype(int)
    window_performance = _base_split_performance(
        subset_meta,
        subset["target"].astype(int),
        "walk_subset_train",
        "window in_sample_after_refit; optimistic",
        config.prediction_threshold,
    )
    window_performance.extend(
        _base_split_performance(
            holdout_meta,
            y_holdout,
            "walk_blend_holdout",
            "window base-model out_of_sample; used to train meta-model",
            config.prediction_threshold,
        )
    )
    class_ratio = float((y_holdout == 0).sum() / max((y_holdout == 1).sum(), 1))
    scale_pos_weight = class_ratio if config.meta_class_weight == "balanced" else 1.0
    meta_model = clone(_xgb_estimator(config)).set_params(
        **tuning["xgboost_meta"]["best_params"],
        scale_pos_weight=scale_pos_weight,
    )
    meta_model.fit(holdout_meta, y_holdout)

    prediction_meta = _probability_frame(base_models, prediction_block[features])
    probability = meta_model.predict_proba(prediction_meta)[:, 1]
    audit = {
        "history_start": history.index.min(),
        "history_end": history.index.max(),
        "history_rows": len(history),
        "subset_start": subset.index.min(),
        "subset_end": subset.index.max(),
        "subset_rows": len(subset),
        "holdout_start": holdout.index.min(),
        "holdout_end": holdout.index.max(),
        "holdout_rows": len(holdout),
        "prediction_start": prediction_block.index.min(),
        "prediction_end": prediction_block.index.max(),
        "prediction_rows": len(prediction_block),
        "meta_scale_pos_weight": scale_pos_weight,
    }
    return prediction_meta, probability, base_models, meta_model, audit, window_performance


def _walk_forward_predict(selected_frame, validation, transformer, features, tuning, config):
    """Create a stitched out-of-sample Final prediction using frozen parameters."""

    meta_blocks, probability_blocks, audits, window_performance_rows = [], [], [], []
    last_base_models, last_meta_model = None, None
    refit_ids = []
    for refit_id, block in enumerate(
        _walk_forward_period_blocks(validation, config.walk_forward_frequency), start=1
    ):
        # Purge the label boundary before each prediction block.
        available_history = selected_frame.loc[selected_frame.index < block.index.min()].copy()
        (
            block_meta, block_probability, last_base_models, last_meta_model,
            audit, window_performance,
        ) = (
            _fit_frozen_blend_window(
                available_history, block, transformer, features, tuning, config
            )
        )
        audit["refit_id"] = refit_id
        audits.append(audit)
        for row in window_performance:
            row["refit_id"] = refit_id
            row["prediction_start"] = block.index.min()
            row["prediction_end"] = block.index.max()
        window_performance_rows.extend(window_performance)
        meta_blocks.append(block_meta)
        probability_blocks.append(pd.Series(block_probability, index=block.index))
        refit_ids.append(pd.Series(refit_id, index=block.index, dtype=int))

    validation_meta = pd.concat(meta_blocks).sort_index().reindex(validation.index)
    probability = pd.concat(probability_blocks).sort_index().reindex(validation.index).to_numpy()
    refit_series = pd.concat(refit_ids).sort_index().reindex(validation.index)
    audit_frame = pd.DataFrame(audits).set_index("refit_id")
    window_performance_frame = pd.DataFrame(window_performance_rows)
    return (
        validation_meta, probability, refit_series, audit_frame,
        window_performance_frame, last_base_models, last_meta_model,
    )


def _classification_metrics(y, probability, threshold):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return prediction, {
        "roc_auc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


def optimize_prediction_threshold(
    meta,
    probability,
    transaction_cost_bps=0.0,
    objective="total_return",
    grid_size=101,
    min_exposure=0.05,
):
    """Choose a trading cutoff from development data without a look-ahead shift.

    A probability observed at close *t* maps directly to ``forward_return``
    from close *t* to close *t+1*.  The production backtest stores that same
    P&L on row *t+1*, hence its explicit one-row position shift.
    """

    if objective not in {"total_return", "sharpe"}:
        raise ValueError("threshold_objective must be 'total_return' or 'sharpe'.")
    if "forward_return" not in meta:
        raise ValueError("Threshold optimization requires forward_return.")
    grid_size = max(11, int(grid_size))
    probability = pd.Series(np.asarray(probability, dtype=float), index=meta.index)
    valid = probability.notna() & meta["forward_return"].notna()
    probability = probability.loc[valid]
    forward_return = meta.loc[valid, "forward_return"].astype(float)
    if probability.empty:
        raise ValueError("No valid development probabilities are available for threshold optimization.")
    candidates = np.unique(np.quantile(probability, np.linspace(0.0, 0.95, grid_size)))
    rows = []
    for threshold in candidates:
        signal = (probability >= threshold).astype(float)
        exposure = float(signal.mean())
        if exposure < float(min_exposure):
            continue
        turnover = signal.diff().abs().fillna(signal.abs())
        daily = signal * forward_return - turnover * float(transaction_cost_bps) / 10_000.0
        equity = (1.0 + daily).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)
        sharpe = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else 0.0
        drawdown = float((equity / equity.cummax() - 1.0).min())
        rows.append(
            {
                "threshold": float(threshold),
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": drawdown,
                "exposure": exposure,
                "turnover": float(turnover.sum()),
            }
        )
    if not rows:
        raise ValueError("No threshold candidate satisfies threshold_min_exposure.")
    ranking = pd.DataFrame(rows).sort_values(
        [objective, "total_return", "sharpe"], ascending=False
    ).reset_index(drop=True)
    return float(ranking.iloc[0]["threshold"]), ranking


def _meta_oof_probability(holdout_meta, y_holdout, tuning, config):
    """Generate strict chronological OOF meta probabilities on Blend Holdout."""

    probability = pd.Series(np.nan, index=holdout_meta.index, dtype=float)
    splitter = TimeSeriesSplit(n_splits=config.meta_cv_splits, gap=1)
    for train_index, valid_index in splitter.split(holdout_meta):
        y_train = y_holdout.iloc[train_index]
        ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        scale_pos_weight = ratio if config.meta_class_weight == "balanced" else 1.0
        model = clone(_xgb_estimator(config)).set_params(
            **tuning["xgboost_meta"]["best_params"],
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(holdout_meta.iloc[train_index], y_train)
        probability.iloc[valid_index] = model.predict_proba(holdout_meta.iloc[valid_index])[:, 1]
    return probability


def _base_split_performance(probability_frame, y, split_name, evaluation_nature, threshold):
    """Build auditable base-learner metrics for one chronological partition."""

    rows = []
    y = pd.Series(y, index=probability_frame.index).astype(int)
    for model_name in ("lr", "svc", "knn"):
        column = f"{model_name}_probability"
        probability = probability_frame[column].to_numpy()
        _, metrics = _classification_metrics(y, probability, threshold)
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "evaluation_nature": evaluation_nature,
                "rows": int(len(y)),
                "target_positive_rate": float(y.mean()),
                "probability_mean": float(np.mean(probability)),
                "probability_std": float(np.std(probability, ddof=1)),
                "predicted_positive_rate": float(np.mean(probability >= threshold)),
                **metrics,
            }
        )
    return rows


def _archive_experiment(summary):
    """Snapshot configuration, tables, figures and fitted models for one run."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    roc_auc = float(summary["metrics"]["roc_auc"])
    strategy_return = 100.0 * float(summary["backtest"]["strategy_total_return"])
    base_name = f"{timestamp}_roc-{roc_auc:.3f}_return-{strategy_return:.2f}pct"
    runs_directory = OUTPUT_DIR / "runs"
    runs_directory.mkdir(parents=True, exist_ok=True)
    run_directory = runs_directory / base_name
    suffix = 1
    while run_directory.exists():
        run_directory = runs_directory / f"{base_name}_{suffix:02d}"
        suffix += 1

    configuration_directory = run_directory / "configuration"
    tables_directory = run_directory / "tables"
    figures_directory = run_directory / "figures"
    models_directory = run_directory / "models"
    for directory in (
        configuration_directory, tables_directory, figures_directory, models_directory
    ):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        run_archive = run_directory.relative_to(ROOT.parent).as_posix()
    except ValueError:
        run_archive = run_directory.resolve().as_posix()
    summary["run_archive"] = run_archive
    pd.Series(summary, dtype=object).to_json(TABLES_DIR / "model_summary.json", indent=2)

    copied = {"configuration": [], "tables": [], "figures": [], "models": []}
    for source in sorted(PROCESSED_DIR.glob("*.json")):
        shutil.copy2(source, configuration_directory / source.name)
        copied["configuration"].append(source.name)
    for source in sorted(TABLES_DIR.iterdir()):
        if source.is_file() and source.suffix.lower() in {".csv", ".json"}:
            shutil.copy2(source, tables_directory / source.name)
            copied["tables"].append(source.name)
    for source in sorted(FIGURES_DIR.iterdir()):
        if source.is_file() and source.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            shutil.copy2(source, figures_directory / source.name)
            copied["figures"].append(source.name)
    for source in sorted(MODELS_DIR.iterdir()):
        if source.is_file() and source.suffix.lower() in {".joblib", ".json"}:
            shutil.copy2(source, models_directory / source.name)
            copied["models"].append(source.name)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folder_naming": "datetime_roc_auc_strategy_return",
        "evaluation_mode": summary.get("evaluation_mode"),
        "roc_auc": roc_auc,
        "strategy_total_return": float(summary["backtest"]["strategy_total_return"]),
        "copied_files": copied,
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_directory


def run_backtest(meta, probability, threshold, transaction_cost_bps):
    frame = meta.copy()
    frame["probability"] = np.asarray(probability)
    frame["raw_signal"] = (frame["probability"] >= threshold).astype(float)
    frame["position"] = frame["raw_signal"].shift(1).fillna(0.0)
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["cost"] = frame["turnover"] * float(transaction_cost_bps) / 10_000.0
    frame["strategy_return"] = frame["position"] * frame["realized_return"].fillna(0.0) - frame["cost"]
    frame["buy_hold_return"] = frame["realized_return"].fillna(0.0)
    frame["strategy_equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["buy_hold_equity"] = (1.0 + frame["buy_hold_return"]).cumprod()
    frame["drawdown"] = frame["strategy_equity"] / frame["strategy_equity"].cummax() - 1.0
    return frame


def _backtest_metrics(frame):
    daily = frame["strategy_return"]
    sharpe = np.sqrt(252) * daily.mean() / daily.std() if daily.std() > 0 else 0.0
    return {
        "strategy_total_return": float(frame["strategy_equity"].iloc[-1] - 1.0),
        "buy_hold_total_return": float(frame["buy_hold_equity"].iloc[-1] - 1.0),
        "strategy_sharpe": float(sharpe),
        "strategy_max_drawdown": float(frame["drawdown"].min()),
        "turnover": float(frame["turnover"].sum()),
        "time_in_market": float(frame["position"].mean()),
    }


def _save_plots(y, probability, prediction, backtest, model_comparison, validation_meta, meta_model):
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    plt.figure(figsize=(6, 5)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--"); plt.title("Final Validation ROC"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 5)); plt.plot(recall, precision); plt.title("Final Validation Precision-Recall"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(5, 4)); sns.heatmap(confusion_matrix(y, prediction), annot=True, fmt="d", cmap="Blues"); plt.title("Final Validation Confusion Matrix"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=180); plt.close()
    plt.figure(figsize=(9, 4)); plt.plot(backtest.index, backtest["strategy_equity"], label="Strategy"); plt.plot(backtest.index, backtest["buy_hold_equity"], label="Buy & Hold"); plt.legend(); plt.title("Final Validation Equity Curve"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "equity_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(9, 3.5)); plt.fill_between(backtest.index, backtest["drawdown"], 0, alpha=0.5); plt.title("Strategy Drawdown"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "drawdown.png", dpi=180); plt.close()
    comparison_plot = model_comparison.set_index("model")[["roc_auc", "balanced_accuracy", "f1"]]
    comparison_plot.plot(kind="bar", figsize=(8, 4)); plt.ylim(0, 1); plt.title("Final Validation Model Comparison"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 5)); sns.heatmap(validation_meta.corr(), annot=True, cmap="coolwarm", center=0); plt.title("Base Probability Correlation"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "meta_feature_correlation.png", dpi=180); plt.close()
    if hasattr(meta_model, "feature_importances_"):
        plt.figure(figsize=(6, 4)); plt.bar(validation_meta.columns, meta_model.feature_importances_, color="darkorange"); plt.title("XGBoost Meta Feature Importance"); plt.xticks(rotation=25, ha="right"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "meta_feature_importance.png", dpi=180); plt.close()


def _development_candidate_score(selected_frame, transformer, config, candidate_id):
    """Evaluate one feature subset without accessing Final Validation labels.

    The 60% SubsetTrain portion is used for base learner tuning; their
    probabilities on the following 20% Blend Holdout form the sole data for
    XGBoost's time-series CV.  The terminal 20% partition is deliberately not
    materialised or scored here.
    """

    split = chronological_blending_split(selected_frame, purge_size=1)
    features = [
        column for column in selected_frame.columns
        if column not in {"target", "forward_return", "realized_return"}
    ]
    subset, holdout = split["subset_train"], split["holdout"]
    X_subset, y_subset = subset[features], subset["target"].astype(int)
    X_holdout, y_holdout = holdout[features], holdout["target"].astype(int)
    grids = {"lr": config.lr_grid, "svc": config.svc_grid, "knn": config.knn_grid}

    tuned_models, tuning = {}, {}
    for name, estimator in _base_estimators(transformer).items():
        tuned_models[name], tuning[name] = _search(
            estimator,
            grids[name],
            X_subset,
            y_subset,
            config,
            name,
            config.base_cv_splits,
            TABLES_DIR / f"auto_candidate_{candidate_id:02d}_tuning_{name}.csv",
        )

    holdout_meta = _probability_frame(tuned_models, X_holdout)
    ratio = float((y_subset == 0).sum() / max((y_subset == 1).sum(), 1))
    scale_pos_weight = ratio if config.meta_class_weight == "balanced" else 1.0
    _, tuning["xgboost_meta"] = _search(
        _xgb_estimator(config).set_params(scale_pos_weight=scale_pos_weight),
        config.meta_grid,
        holdout_meta,
        y_holdout,
        config,
        "xgboost_meta",
        config.meta_cv_splits,
        TABLES_DIR / f"auto_candidate_{candidate_id:02d}_tuning_xgboost_meta.csv",
    )
    meta_oof = _meta_oof_probability(holdout_meta, y_holdout, tuning, config)
    threshold, threshold_ranking = optimize_prediction_threshold(
        holdout[["forward_return"]],
        meta_oof,
        transaction_cost_bps=config.transaction_cost_bps,
        objective=config.threshold_objective,
        grid_size=config.threshold_grid_size,
        min_exposure=config.threshold_min_exposure,
    )
    threshold_best = threshold_ranking.iloc[0]
    base_scores = [
        result["best_score"] for result in tuning.values()
        if result.get("best_score") is not None and np.isfinite(result["best_score"])
    ]
    return {
        "candidate_id": int(candidate_id),
        "feature_count": int(len(features)),
        "feature_names": features,
        "development_metric": config.search_metric,
        "meta_cv_score": float(tuning["xgboost_meta"]["best_score"]),
        "base_cv_score_mean": float(np.mean(base_scores[:3])),
        "lr_cv_score": float(tuning["lr"]["best_score"]),
        "svc_cv_score": float(tuning["svc"]["best_score"]),
        "knn_cv_score": float(tuning["knn"]["best_score"]),
        "meta_cv_splits": int(config.meta_cv_splits),
        "base_cv_splits": int(config.base_cv_splits),
        "development_threshold": float(threshold),
        "development_strategy_return": float(threshold_best["total_return"]),
        "development_strategy_sharpe": float(threshold_best["sharpe"]),
        "development_strategy_exposure": float(threshold_best["exposure"]),
        "tuning": tuning,
        "final_test_labels_used": False,
    }


def optimize_blending_development(
    feature_frame,
    eda_frame,
    ranked_features,
    config=None,
    feature_counts=(20, 30, 40, 48),
):
    """Select a Blending specification using development data only.

    It searches nested feature subsets ordered by the train-only embedded
    ranking, then selects the specification with the strongest XGBoost
    Holdout time-series-CV score.  Ties favour higher average base CV score
    and fewer features.  The returned frame can subsequently be sent through
    :func:`train_blending_ensemble`, which is the *only* function that reads
    Final Validation labels.
    """

    ensure_directories()
    config = config or ModelConfig()
    if not config.auto_tune:
        raise ValueError("Development auto-optimization requires 'auto_tune=True'.")
    available = [
        feature for feature in ranked_features
        if feature in feature_frame.columns and feature in set(eda_frame["feature"])
    ]
    if not available:
        raise ValueError("No ranked, train-only selected features are available for auto-optimization.")
    requested_counts = sorted({int(count) for count in feature_counts if int(count) > 0})
    effective_counts = sorted({min(count, len(available)) for count in requested_counts})
    if not effective_counts:
        raise ValueError("At least one positive feature-subset size is required.")

    candidate_rows, candidates = [], []
    for candidate_id, count in enumerate(effective_counts, start=1):
        features = available[:count]
        candidate_frame = feature_frame[
            [*features, "target", "forward_return", "realized_return"]
        ].copy()
        from feature_selection import build_column_transformer

        transformer, _ = build_column_transformer(eda_frame, features)
        result = _development_candidate_score(
            candidate_frame, transformer, config, candidate_id
        )
        candidate_rows.append(
            {key: value for key, value in result.items() if key not in {"feature_names", "tuning"}}
        )
        candidates.append((result, candidate_frame, transformer))

    ranking = pd.DataFrame(candidate_rows).sort_values(
        ["meta_cv_score", "development_strategy_return", "base_cv_score_mean", "feature_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "development_rank", np.arange(1, len(ranking) + 1))
    winning_id = int(ranking.iloc[0]["candidate_id"])
    winner, winner_frame, winner_transformer = next(
        candidate for candidate in candidates if candidate[0]["candidate_id"] == winning_id
    )
    ranking.to_csv(TABLES_DIR / "auto_development_optimization_candidates.csv", index=False)
    selection = {
        "selection_scope": "SubsetTrain + Blend Holdout chronological CV only",
        "selection_metric": config.search_metric,
        "selection_rule": "max meta_cv_score, then max development strategy return, mean base CV score, and fewer features",
        "final_test_labels_used_for_selection": False,
        "candidate_feature_counts": effective_counts,
        "winner": winner,
    }
    (TABLES_DIR / "auto_development_optimization_best.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return winner_frame, winner_transformer, selection, ranking


def train_blending_ensemble(selected_frame, transformer, config=None):
    """Train bases on 60%, XGBoost meta on next 20%, evaluate final 20% once."""
    ensure_directories()
    config = config or ModelConfig()
    split = chronological_blending_split(selected_frame, purge_size=1)
    features = [column for column in selected_frame.columns if column not in {"target", "forward_return", "realized_return"}]
    subset, holdout, validation = split["subset_train"], split["holdout"], split["final_validation"]
    X_subset, y_subset = subset[features], subset["target"].astype(int)
    X_holdout, y_holdout = holdout[features], holdout["target"].astype(int)
    X_validation, y_validation = validation[features], validation["target"].astype(int)

    def class_distribution(y):
        counts = y.value_counts().reindex([0, 1], fill_value=0)
        return {
            "class_0": int(counts.loc[0]),
            "class_1": int(counts.loc[1]),
            "positive_rate": float(counts.loc[1] / max(len(y), 1)),
            "negative_to_positive_ratio": float(counts.loc[0] / max(counts.loc[1], 1)),
        }

    grids = {"lr": config.lr_grid, "svc": config.svc_grid, "knn": config.knn_grid}
    tuned_models, tuning = {}, {}
    for name, estimator in _base_estimators(transformer).items():
        tuned_models[name], tuning[name] = _search(estimator, grids[name], X_subset, y_subset, config, name, config.base_cv_splits)

    holdout_meta = _probability_frame(tuned_models, X_holdout)
    subset_meta = _probability_frame(tuned_models, X_subset)
    base_split_performance = _base_split_performance(
        subset_meta,
        y_subset,
        "subset_train",
        "in_sample_after_refit; optimistic",
        config.prediction_threshold,
    )
    base_split_performance.extend(
        _base_split_performance(
            holdout_meta,
            y_holdout,
            "blend_holdout",
            "base-model out_of_sample; used to train meta-model",
            config.prediction_threshold,
        )
    )
    # Use the earlier SubsetTrain class ratio during meta-model CV.
    raw_meta_tuning_class_ratio = float((y_subset == 0).sum() / max((y_subset == 1).sum(), 1))
    meta_tuning_class_ratio = raw_meta_tuning_class_ratio if config.meta_class_weight == "balanced" else 1.0
    meta_model, tuning["xgboost_meta"] = _search(
        _xgb_estimator(config).set_params(scale_pos_weight=meta_tuning_class_ratio),
        config.meta_grid, holdout_meta, y_holdout, config, "xgboost_meta", config.meta_cv_splits
    )
    threshold_optimization = {
        "enabled": bool(config.optimize_threshold),
        "selection_scope": "Blend Holdout chronological meta OOF only" if config.optimize_threshold else "fixed user setting",
        "objective": config.threshold_objective if config.optimize_threshold else None,
        "final_test_labels_used": False,
    }
    if config.optimize_threshold:
        meta_oof = _meta_oof_probability(holdout_meta, y_holdout, tuning, config)
        effective_threshold, threshold_ranking = optimize_prediction_threshold(
            holdout[["forward_return"]],
            meta_oof,
            transaction_cost_bps=config.transaction_cost_bps,
            objective=config.threshold_objective,
            grid_size=config.threshold_grid_size,
            min_exposure=config.threshold_min_exposure,
        )
        threshold_ranking.to_csv(TABLES_DIR / "threshold_optimization.csv", index=False)
        threshold_optimization.update(
            {
                "selected_threshold": float(effective_threshold),
                "candidate_count": int(len(threshold_ranking)),
                "development_metrics": threshold_ranking.iloc[0].to_dict(),
                "table": "threshold_optimization.csv",
            }
        )
        config.prediction_threshold = float(effective_threshold)
        base_split_performance = _base_split_performance(
            subset_meta,
            y_subset,
            "subset_train",
            "in_sample_after_refit; optimistic",
            config.prediction_threshold,
        )
        base_split_performance.extend(
            _base_split_performance(
                holdout_meta,
                y_holdout,
                "blend_holdout",
                "base-model out_of_sample; used to train meta-model",
                config.prediction_threshold,
            )
        )
    else:
        threshold_optimization["selected_threshold"] = float(config.prediction_threshold)
    # Refit the frozen meta-model with the complete Holdout class ratio.
    raw_meta_final_class_ratio = float((y_holdout == 0).sum() / max((y_holdout == 1).sum(), 1))
    meta_final_class_ratio = raw_meta_final_class_ratio if config.meta_class_weight == "balanced" else 1.0
    meta_model.set_params(scale_pos_weight=meta_final_class_ratio)
    meta_model.fit(holdout_meta, y_holdout)
    tuning["xgboost_meta"].update(
        {
            "cv_scale_pos_weight": meta_tuning_class_ratio,
            "configured_class_weight": config.meta_class_weight,
            "cv_weight_source": "earlier SubsetTrain labels" if config.meta_class_weight == "balanced" else "disabled (1.0)",
            "final_scale_pos_weight": meta_final_class_ratio,
            "final_weight_source": "full Holdout after hyperparameters were frozen" if config.meta_class_weight == "balanced" else "disabled (1.0)",
        }
    )
    refit_series = None
    walk_forward_audit = None
    walk_forward_base_performance = None
    if config.evaluation_mode == "walk_forward":
        (
            validation_meta,
            probability,
            refit_series,
            walk_forward_audit,
            walk_forward_base_performance,
            final_base_models,
            final_meta_model,
        ) = _walk_forward_predict(
            selected_frame, validation, transformer, features, tuning, config
        )
    elif config.evaluation_mode == "static":
        validation_meta = _probability_frame(tuned_models, X_validation)
        probability = meta_model.predict_proba(validation_meta)[:, 1]
        final_base_models, final_meta_model = tuned_models, meta_model
    else:
        raise ValueError("evaluation_mode must be 'static' or 'walk_forward'.")
    base_split_performance.extend(
        _base_split_performance(
            validation_meta,
            y_validation,
            "final_test",
            "stitched walk_forward out_of_sample" if config.evaluation_mode == "walk_forward"
            else "single untouched out_of_sample",
            config.prediction_threshold,
        )
    )
    base_split_performance_frame = pd.DataFrame(base_split_performance)
    prediction, metrics = _classification_metrics(y_validation, probability, config.prediction_threshold)
    comparison_rows = []
    for name, model in tuned_models.items():
        base_probability = validation_meta[f"{name}_probability"].to_numpy()
        _, base_metrics = _classification_metrics(y_validation, base_probability, config.prediction_threshold)
        comparison_rows.append({"model": name, **base_metrics})
    comparison_rows.append({"model": "xgboost_blending", **metrics})
    model_comparison = pd.DataFrame(comparison_rows)
    backtest = run_backtest(
        validation[["realized_return", "forward_return"]], probability,
        config.prediction_threshold, config.transaction_cost_bps,
    )
    backtest_summary = _backtest_metrics(backtest)
    report = classification_report(y_validation, prediction, output_dict=True, zero_division=0)

    holdout_meta.assign(target=y_holdout).to_csv(TABLES_DIR / "holdout_meta_training_data.csv")
    prediction_details = validation_meta.assign(
        probability=probability, prediction=prediction, target=y_validation
    )
    if refit_series is not None:
        prediction_details["refit_id"] = refit_series.astype(int)
        walk_forward_audit.to_csv(TABLES_DIR / "walk_forward_refits.csv")
        walk_forward_base_performance.to_csv(
            TABLES_DIR / "walk_forward_base_split_performance.csv", index=False
        )
    prediction_details.to_csv(TABLES_DIR / "final_validation_predictions.csv")
    backtest.to_csv(TABLES_DIR / "final_validation_backtest.csv")
    pd.DataFrame(confusion_matrix(y_validation, prediction)).to_csv(TABLES_DIR / "confusion_matrix.csv", index=False)
    pd.DataFrame(report).T.to_csv(TABLES_DIR / "classification_report.csv")
    model_comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    base_split_performance_frame.to_csv(
        TABLES_DIR / "base_model_split_performance.csv", index=False
    )
    _save_plots(
        y_validation, probability, prediction, backtest,
        model_comparison, validation_meta, final_meta_model,
    )

    bundle = {
        "features": features, "base_models": final_base_models, "meta_model": final_meta_model,
        "transformer": transformer, "prediction_threshold": config.prediction_threshold,
        "evaluation_mode": config.evaluation_mode,
    }
    joblib.dump(bundle, MODELS_DIR / "blending_lr_svc_knn_xgb.joblib")
    summary = {
        "architecture": {"base_learners": ["lr", "svc", "knn"], "meta_model": "xgboost"},
        "evaluation_mode": config.evaluation_mode,
        "walk_forward": {
            "enabled": config.evaluation_mode == "walk_forward",
            "frequency": config.walk_forward_frequency if config.evaluation_mode == "walk_forward" else None,
            "hyperparameters_frozen_after_initial_60_20_development_search": config.evaluation_mode == "walk_forward",
            "refit_count": int(len(walk_forward_audit)) if walk_forward_audit is not None else 0,
            "historical_refit_split": "75% base SubsetTrain / 25% Blend Holdout" if config.evaluation_mode == "walk_forward" else None,
            "future_labels_used_for_each_prediction": False,
            "boundary_purge_rows": 1 if config.evaluation_mode == "walk_forward" else None,
            "per_window_base_performance_table": "walk_forward_base_split_performance.csv"
            if config.evaluation_mode == "walk_forward" else None,
        },
        "meta_probability_constraints": {
            "monotone_in_each_base_probability": config.meta_monotone_probabilities,
            "constraints": [1, 1, 1] if config.meta_monotone_probabilities else None,
            "class_weight": config.meta_class_weight,
        },
        "nominal_split": "60% SubsetTrain / 20% Holdout / 20% Final Validation",
        "tuning_cv": {
            "base_models": config.base_cv_splits, "xgboost_meta": config.meta_cv_splits, "gap": 1,
            "base_search_budget": config.base_search_iterations or config.search_iterations,
            "meta_search_budget": config.meta_search_iterations or config.search_iterations,
        },
        "effective_rows": {"subset_train": len(subset), "holdout": len(holdout), "final_validation": len(validation)},
        "periods": {
            name: {"start": str(frame.index.min().date()), "end": str(frame.index.max().date()), "rows": len(frame)}
            for name, frame in (("subset_train", subset), ("holdout", holdout), ("final_validation", validation))
        },
        "test_or_validation_labels_used_for_tuning": False,
        "class_imbalance": {
            "distributions": {
                "subset_train": class_distribution(y_subset),
                "holdout": class_distribution(y_holdout),
                "final_validation": class_distribution(y_validation),
            },
            "lr": {
                "selected_class_weight": tuning["lr"]["best_params"].get("model__class_weight"),
                "selection_scope": "SubsetTrain TimeSeriesSplit only",
            },
            "svc": {
                "selected_class_weight": tuning["svc"]["best_params"].get("model__class_weight"),
                "selection_scope": "SubsetTrain TimeSeriesSplit only",
            },
            "knn": "no native class_weight support; scaling and imbalance-aware evaluation are used",
            "xgboost_meta": {
                "cv_scale_pos_weight": meta_tuning_class_ratio,
                "unweighted_subset_ratio": raw_meta_tuning_class_ratio,
                "cv_weight_source": "earlier SubsetTrain labels" if config.meta_class_weight == "balanced" else "disabled (1.0)",
                "final_scale_pos_weight": meta_final_class_ratio,
                "unweighted_holdout_ratio": raw_meta_final_class_ratio,
                "final_weight_source": "full Holdout after hyperparameters were frozen" if config.meta_class_weight == "balanced" else "disabled (1.0)",
            },
            "synthetic_resampling": False,
            "reason_no_smote": "Synthetic/random resampling can distort chronological dependence in financial data.",
            "imbalance_aware_metrics": ["balanced_accuracy", "precision", "recall", "f1", "roc_auc"],
        },
        "tuning": tuning,
        "auto_development_optimization": config.auto_optimization_details,
        "threshold_optimization": threshold_optimization,
        "metrics": metrics,
        "model_comparison": comparison_rows,
        "base_model_split_performance": base_split_performance,
        "classification_report": report,
        "backtest": backtest_summary,
        "prediction_threshold": config.prediction_threshold,
        "transaction_cost_bps": config.transaction_cost_bps,
    }
    pd.Series(summary, dtype=object).to_json(TABLES_DIR / "model_summary.json", indent=2)
    _archive_experiment(summary)
    return summary

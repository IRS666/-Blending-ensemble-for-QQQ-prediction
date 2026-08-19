"""Train-only EDA, heterogeneous transformation, and three-stage selection."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from nvconfig import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, SEED, TABLES_DIR, ensure_directories


META_COLUMNS = ["target", "forward_return", "realized_return"]
METRIC_NAMES = ("roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1")
SELECTION_STATE_PATH = PROCESSED_DIR / "feature_selection_state.joblib"


@dataclass
class SelectionConfig:
    missing_threshold: float = 0.20
    minimum_abs_correlation: float = 0.015
    minimum_mi_quantile: float = 0.50
    stage1_max_features: int = 400
    stage2_metric: str = "roc_auc"
    stage2_min_metric: float = 0.0
    stage2_top_features: int = 160
    multicollinearity_threshold: float = 0.92
    embedded_method: str = "extra_trees"
    embedded_top_features: int = 80
    rfe_enabled: bool = True
    rfe_trigger: int = 60
    final_feature_count: int = 40
    cv_splits: int = 4
    n_jobs: int = -1


def _safe_auc(y_true, probability):
    return roc_auc_score(y_true, probability) if pd.Series(y_true).nunique() > 1 else np.nan


def _effective_n_jobs(requested_jobs, task_count=None):
    """Bound automatic parallelism to avoid nested over-subscription."""
    if task_count is not None and task_count <= 1:
        return 1
    available = os.cpu_count() or 1
    requested = min(8, available) if requested_jobs == -1 else requested_jobs
    if requested <= 0:
        raise ValueError("Feature-selection n_jobs must be -1 (automatic) or a positive integer.")
    return max(1, min(int(requested), int(task_count))) if task_count is not None else max(1, int(requested))


def build_feature_eda(X_train, y_train, n_jobs=-1):
    """Create one row of distribution and target-association diagnostics per feature."""
    numeric = X_train.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    median_filled = numeric.fillna(numeric.median()).fillna(0.0)
    mi = pd.Series(
        mutual_info_classif(
            median_filled,
            y_train,
            random_state=SEED,
            n_jobs=_effective_n_jobs(n_jobs, numeric.shape[1]),
        ),
        index=numeric.columns,
    )
    rows = []
    for feature in numeric.columns:
        series = numeric[feature]
        finite = series.dropna()
        q1, q3 = finite.quantile([0.25, 0.75]) if not finite.empty else (np.nan, np.nan)
        iqr = q3 - q1
        outlier_ratio = float(((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
        correlation = float(series.corr(y_train)) if finite.nunique() > 1 else 0.0
        skew = float(finite.skew()) if len(finite) > 2 else 0.0
        scale_strategy = (
            "robust"
            if abs(skew) > 2.0 or outlier_ratio > 0.05
            else "minmax"
            if finite.min() >= 0 and finite.max() <= 1
            else "standard"
            if finite.nunique() > 2
            else "passthrough"
        )
        rows.append(
            {
                "feature": feature,
                "count": int(finite.count()),
                "missing_ratio": float(series.isna().mean()),
                "unique_count": int(finite.nunique()),
                "mean": float(finite.mean()) if not finite.empty else np.nan,
                "std": float(finite.std()) if not finite.empty else np.nan,
                "median": float(finite.median()) if not finite.empty else np.nan,
                "minimum": float(finite.min()) if not finite.empty else np.nan,
                "maximum": float(finite.max()) if not finite.empty else np.nan,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "skewness": skew,
                "kurtosis": float(finite.kurt()) if len(finite) > 3 else 0.0,
                "outlier_ratio_iqr": outlier_ratio,
                "target_correlation": correlation,
                "abs_target_correlation": abs(correlation),
                "mutual_information": float(mi.get(feature, 0.0)),
                "scale_strategy": scale_strategy,
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def build_column_transformer(eda_frame, selected_features):
    """Build per-feature transformations chosen from Train-only EDA."""
    selected = set(selected_features)
    groups = {
        name: eda_frame.loc[(eda_frame["scale_strategy"] == name) & eda_frame["feature"].isin(selected), "feature"].tolist()
        for name in ("robust", "minmax", "standard", "passthrough")
    }
    transformers = []
    scaler_map = {
        "robust": RobustScaler(quantile_range=(25, 75)),
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "passthrough": "passthrough",
    }
    for name, columns in groups.items():
        if not columns:
            continue
        scaler = scaler_map[name]
        if scaler == "passthrough":
            transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
        else:
            transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)])
        transformers.append((name, transformer, columns))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False), groups


def _clean_from_eda(eda, config):
    return eda.loc[
        (eda["missing_ratio"] <= config.missing_threshold)
        & (eda["unique_count"] > 1)
        & eda["std"].fillna(0.0).gt(0.0)
    ].copy()


def _stage_snapshot(
    features,
    eda,
    stage_name,
    metric_scores=None,
    embedded_ranking=None,
    rfe_ranking=None,
):
    """Build an auditable table for the features retained after one stage.

    Rows follow the exact current ranking.  Metrics become cumulative: EDA and
    target-association diagnostics are available from the beginning; temporal
    univariate classification metrics, embedded importance and RFE diagnostics
    are appended once those stages have actually been fitted.
    """
    ordered = list(dict.fromkeys(features))
    snapshot = pd.DataFrame(
        {
            "stage": stage_name,
            "stage_rank": np.arange(1, len(ordered) + 1),
            "feature": ordered,
        }
    )
    snapshot = snapshot.merge(eda, on="feature", how="left", validate="one_to_one")
    for extra in (metric_scores, embedded_ranking, rfe_ranking):
        if extra is None or extra.empty:
            continue
        extra_columns = [column for column in extra.columns if column != "feature" and column not in snapshot.columns]
        snapshot = snapshot.merge(
            extra[["feature", *extra_columns]],
            on="feature",
            how="left",
            validate="one_to_one",
        )
    return snapshot.sort_values("stage_rank").reset_index(drop=True)


def _feature_lifecycle(initial_features, stages):
    """Create one-row-per-feature lineage across all selection stages."""
    lifecycle = pd.DataFrame({"feature": list(initial_features)})
    previous_present = pd.Series(True, index=lifecycle.index)
    first_removed = pd.Series("retained_final", index=lifecycle.index, dtype=object)
    for stage_id, stage_label, features in stages:
        rank_map = {feature: rank for rank, feature in enumerate(features, start=1)}
        rank_column = f"rank_{stage_id}"
        retained_column = f"retained_{stage_id}"
        lifecycle[rank_column] = lifecycle["feature"].map(rank_map).astype("Int64")
        lifecycle[retained_column] = lifecycle[rank_column].notna()
        just_removed = previous_present & ~lifecycle[retained_column]
        first_removed.loc[just_removed] = stage_label
        previous_present = lifecycle[retained_column]
    lifecycle["first_removed_stage"] = first_removed
    lifecycle["retained_final"] = previous_present
    return lifecycle


def hourglass_filter(eda, config):
    """Stage 1: wide-to-narrow correlation/MI funnel fitted on SubsetTrain only."""
    clean = _clean_from_eda(eda, config)
    positive_mi = clean.loc[clean["mutual_information"] > 0, "mutual_information"].dropna()
    mi_threshold = float(positive_mi.quantile(config.minimum_mi_quantile)) if not positive_mi.empty else 0.0
    clean["stage1_score"] = clean["abs_target_correlation"] + clean["mutual_information"]
    selected = clean.loc[
        (clean["abs_target_correlation"] >= config.minimum_abs_correlation)
        | ((clean["mutual_information"] > 0) & (clean["mutual_information"] >= mi_threshold))
    ].sort_values(["stage1_score", "abs_target_correlation"], ascending=False)
    return selected.head(config.stage1_max_features).copy(), mi_threshold


def _univariate_feature_metrics(feature, series, y, cv_splits):
    """Score one feature across chronological folds."""
    splitter = TimeSeriesSplit(n_splits=cv_splits, gap=1)
    X_feature = series.to_frame(name=feature)
    fold_metrics = []
    for train_index, valid_index in splitter.split(X_feature):
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        if y_train.nunique() < 2 or y_valid.nunique() < 2:
            continue
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
            ]
        )
        pipeline.fit(X_feature.iloc[train_index], y_train)
        probability = pipeline.predict_proba(X_feature.iloc[valid_index])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        fold_metrics.append(
            {
                "roc_auc": _safe_auc(y_valid, probability),
                "accuracy": accuracy_score(y_valid, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_valid, prediction),
                "precision": precision_score(y_valid, prediction, zero_division=0),
                "recall": recall_score(y_valid, prediction, zero_division=0),
                "f1": f1_score(y_valid, prediction, zero_division=0),
            }
        )
    row = {"feature": feature, "folds_used": len(fold_metrics)}
    for metric in METRIC_NAMES:
        row[metric] = float(np.nanmean([fold[metric] for fold in fold_metrics])) if fold_metrics else np.nan
    return row


def _univariate_fold_metrics(feature_frame, y, cv_splits, n_jobs=-1):
    """Run one-feature chronological scores concurrently."""
    effective_jobs = _effective_n_jobs(n_jobs, feature_frame.shape[1])
    rows = joblib.Parallel(n_jobs=effective_jobs, prefer="threads")(
        joblib.delayed(_univariate_feature_metrics)(feature, feature_frame[feature], y, cv_splits)
        for feature in feature_frame.columns
    )
    return pd.DataFrame(rows)


def metric_filter(X_train, y_train, candidates, config):
    """Stage 2: rank candidates using chronological univariate classification metrics."""
    if config.stage2_metric not in METRIC_NAMES:
        raise ValueError(f"Unsupported stage-two metric: {config.stage2_metric}")
    if not 0.0 <= config.stage2_min_metric <= 1.0:
        raise ValueError("Stage-two minimum metric must be between 0 and 1.")
    scores = _univariate_fold_metrics(X_train[candidates], y_train, config.cv_splits, config.n_jobs)
    scores = scores.sort_values([config.stage2_metric, "roc_auc"], ascending=False, na_position="last")
    # Zero ranks all features; a positive threshold filters before ranking.
    scores["passes_stage2_threshold"] = (
        scores[config.stage2_metric].notna()
        if config.stage2_min_metric == 0
        else scores[config.stage2_metric].gt(config.stage2_min_metric)
    )
    eligible = scores.loc[scores["passes_stage2_threshold"]].copy()
    if eligible.empty:
        raise ValueError(
            f"No feature achieved {config.stage2_metric} > {config.stage2_min_metric:.3f} "
            "on chronological SubsetTrain validation folds. Lower the threshold or change the metric."
        )
    return eligible.head(config.stage2_top_features).copy(), scores


def prune_multicollinearity(X_train, ranked_features, rank_scores, threshold):
    """Greedily retain better-ranked features while pruning high correlations."""
    ordered = [feature for feature in ranked_features if feature in X_train]
    correlation = X_train[ordered].corr().abs().fillna(0.0)
    kept, removed = [], []
    for feature in ordered:
        conflicts = [kept_feature for kept_feature in kept if correlation.loc[feature, kept_feature] >= threshold]
        if conflicts:
            removed.append(
                {
                    "feature": feature,
                    "kept_feature": conflicts[0],
                    "absolute_correlation": float(correlation.loc[feature, conflicts[0]]),
                    "rank_score": float(rank_scores.get(feature, np.nan)),
                }
            )
        else:
            kept.append(feature)
    return kept, pd.DataFrame(removed)


def embedded_filter(X_train, y_train, eda, candidates, config):
    """Stage 3: embedded selection fitted exclusively on SubsetTrain.

    ``lasso`` treats the binary label as a numeric 0/1 response and uses the
    absolute LassoCV coefficient solely as a feature-importance score.  It is
    therefore a screening model rather than the project's final classifier.
    """
    transformer, groups = build_column_transformer(eda, candidates)
    transformed = transformer.fit_transform(X_train[candidates])
    transformed_features = transformer.get_feature_names_out().tolist()
    method_details = {"method": config.embedded_method}
    if config.embedded_method == "l1_logistic":
        estimator = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1, class_weight="balanced",
            max_iter=2000, random_state=SEED,
        )
        estimator.fit(transformed, y_train)
        importance = np.abs(estimator.coef_[0])
        method_details.update({"penalty": "l1", "C": 0.1})
    elif config.embedded_method == "lasso":
        estimator = LassoCV(
            alphas=100,
            cv=TimeSeriesSplit(n_splits=config.cv_splits, gap=1),
            max_iter=5000,
            n_jobs=_effective_n_jobs(config.n_jobs, len(candidates)),
            random_state=SEED,
        )
        class_counts = y_train.value_counts()
        sample_weight = y_train.map(
            {label: len(y_train) / (len(class_counts) * count) for label, count in class_counts.items()}
        ).to_numpy()
        estimator.fit(transformed, y_train.to_numpy(dtype=float), sample_weight=sample_weight)
        importance = np.abs(estimator.coef_)
        method_details.update(
            {
                "objective": "weighted_squared_error_plus_l1_penalty",
                "selected_alpha": float(estimator.alpha_),
                "nonzero_coefficients": int(np.count_nonzero(estimator.coef_)),
                "cv": f"TimeSeriesSplit(n_splits={config.cv_splits}, gap=1)",
            }
        )
    elif config.embedded_method == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", n_jobs=_effective_n_jobs(config.n_jobs, len(candidates)), random_state=SEED,
        )
        estimator.fit(transformed, y_train)
        importance = estimator.feature_importances_
        method_details.update({"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 4})
    else:
        raise ValueError(f"Unsupported embedded method: {config.embedded_method}")
    ranking = pd.DataFrame(
        {
            "feature": transformed_features,
            "embedded_importance": importance,
            "embedded_method": config.embedded_method,
        }
    ).sort_values(
        "embedded_importance", ascending=False
    )
    if config.embedded_method == "lasso":
        # Keep one fallback feature for degenerate samples.
        nonzero = ranking.loc[ranking["embedded_importance"] > 1e-12]
        selected_frame = nonzero if not nonzero.empty else ranking.head(1)
    else:
        selected_frame = ranking
    selected = selected_frame.head(config.embedded_top_features)["feature"].tolist()
    return selected, ranking, transformer, groups, method_details


def apply_optional_rfe(X_train, y_train, eda, candidates, config):
    """Use sklearn RFE only when the embedded candidate dimension remains high."""
    if not config.rfe_enabled or len(candidates) <= config.rfe_trigger:
        return candidates[: config.final_feature_count], pd.DataFrame(
            {"feature": candidates, "rfe_rank": np.arange(1, len(candidates) + 1), "rfe_applied": False}
        )
    transformer, _ = build_column_transformer(eda, candidates)
    transformed = transformer.fit_transform(X_train[candidates])
    transformed_features = transformer.get_feature_names_out().tolist()
    estimator = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1500, random_state=SEED)
    selector = RFE(estimator, n_features_to_select=min(config.final_feature_count, len(candidates)), step=0.10)
    selector.fit(transformed, y_train)
    ranking = pd.DataFrame(
        {"feature": transformed_features, "rfe_rank": selector.ranking_, "rfe_selected": selector.support_, "rfe_applied": True}
    ).sort_values(["rfe_rank", "feature"])
    return ranking.loc[ranking["rfe_selected"], "feature"].tolist(), ranking


def _inputs(feature_frame, subset_index):
    X_all = feature_frame.drop(columns=META_COLUMNS, errors="ignore").select_dtypes(include=["number"])
    y_all = feature_frame["target"].astype(int)
    return X_all, y_all, X_all.loc[subset_index], y_all.loc[subset_index]


def _context_key(feature_frame, subset_index):
    payload = pd.util.hash_pandas_object(
        feature_frame.loc[:, feature_frame.columns], index=True
    ).values.tobytes() + pd.util.hash_pandas_object(pd.Index(subset_index)).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _save_state(state):
    joblib.dump(state, SELECTION_STATE_PATH)


def _load_state(feature_frame, subset_index, required):
    if not SELECTION_STATE_PATH.exists():
        raise ValueError("Stage 1 is incomplete. Run EDA and hourglass selection first.")
    state = joblib.load(SELECTION_STATE_PATH)
    if state.get("context_key") != _context_key(feature_frame, subset_index):
        raise ValueError(
            "The feature universe has changed. Restart from Stage 1 to avoid mixing "
            "stale intermediate results."
        )
    if state.get("completed", 0) < required:
        raise ValueError(f"Complete Stage {required} before continuing.")
    return state


def _write_state_outputs(state, feature_frame, subset_index, config):
    """Write cumulative stage tables and a concise resumable-workflow summary."""
    X_all, _, X_train, _ = _inputs(feature_frame, subset_index)
    eda = state["eda"]
    stages = [("quality", "01_quality_clean", state["quality_features"]), ("hourglass", "02_hourglass", state["stage1"]["feature"].tolist())]
    snapshots = {
        "00_initial": _stage_snapshot(state["initial_ranked"], eda, "00_initial"),
        "01_quality_clean": _stage_snapshot(state["quality_features"], eda, "01_quality_clean"),
        "02_hourglass": _stage_snapshot(state["stage1"]["feature"].tolist(), eda, "02_hourglass", metric_scores=state["stage1"]),
    }
    if state.get("completed", 0) >= 2:
        stages.append(("metric", "03_metric", state["stage2_top"]["feature"].tolist()))
        snapshots["03_metric"] = _stage_snapshot(state["stage2_top"]["feature"].tolist(), eda, "03_metric", metric_scores=state["stage2_all"])
    if state.get("completed", 0) >= 3:
        stages.extend([("noncollinear", "04_noncollinear", state["noncollinear"]), ("embedded", "05_embedded", state["embedded_selected"]), ("final", "06_final", state["final_features"])])
        snapshots.update({
            "04_noncollinear": _stage_snapshot(state["noncollinear"], eda, "04_noncollinear", metric_scores=state["stage2_all"]),
            "05_embedded": _stage_snapshot(state["embedded_selected"], eda, "05_embedded", metric_scores=state["stage2_all"], embedded_ranking=state["embedded_ranking"]),
            "06_final": _stage_snapshot(state["final_features"], eda, "06_final", metric_scores=state["stage2_all"], embedded_ranking=state["embedded_ranking"], rfe_ranking=state["rfe_ranking"]),
        })
    for stage_id, snapshot in snapshots.items(): snapshot.to_csv(TABLES_DIR / f"feature_stage_{stage_id}.csv", index=False)
    _feature_lifecycle(state["initial_ranked"], stages).to_csv(TABLES_DIR / "feature_selection_lifecycle.csv", index=False)
    eda.to_csv(TABLES_DIR / "feature_eda_all.csv", index=False)
    state["stage1"].to_csv(TABLES_DIR / "selection_stage1_hourglass.csv", index=False)
    summary = {
        "completed_stage": int(state["completed"]), "initial_features": int(X_all.shape[1]),
        "quality_clean_features": len(state["quality_features"]), "subset_train_rows_used": len(X_train),
        "effective_n_jobs": _effective_n_jobs(config.n_jobs, X_train.shape[1]), "holdout_or_validation_labels_used": False,
        "stage1_mi_threshold": state["mi_threshold"], "stage1_features": len(state["stage1"]),
        "stage2_metric": config.stage2_metric, "stage2_min_metric": float(config.stage2_min_metric),
        "stage2_eligible_features": int(state.get("stage2_all", pd.DataFrame()).get("passes_stage2_threshold", pd.Series(dtype=bool)).sum()),
        "stage2_features": len(state.get("stage2_top", [])), "after_multicollinearity": len(state.get("noncollinear", [])),
        "embedded_method": config.embedded_method, "embedded_features": len(state.get("embedded_selected", [])),
        "final_features": len(state.get("final_features", [])), "final_dataset_ready": state.get("completed", 0) == 3,
        "stage_snapshot_files": {key: f"feature_stage_{key}.csv" for key in snapshots},
    }
    pd.Series(summary, dtype=object).to_json(TABLES_DIR / "feature_selection_summary.json", indent=2)
    return summary


def run_feature_selection_stage1(feature_frame, subset_index, config=None):
    """Fit train-only EDA and the hourglass filter, then persist resumable state."""
    ensure_directories(); config = config or SelectionConfig()
    _, _, X_train, y_train = _inputs(feature_frame, subset_index)
    eda = build_feature_eda(X_train, y_train, _effective_n_jobs(config.n_jobs, X_train.shape[1]))
    initial_ranked = eda.sort_values(["mutual_information", "abs_target_correlation"], ascending=False)["feature"].tolist()
    quality_features = _clean_from_eda(eda, config).sort_values(["mutual_information", "abs_target_correlation"], ascending=False)["feature"].tolist()
    stage1, mi_threshold = hourglass_filter(eda, config)
    state = {"context_key": _context_key(feature_frame, subset_index), "completed": 1, "eda": eda, "initial_ranked": initial_ranked, "quality_features": quality_features, "stage1": stage1, "mi_threshold": mi_threshold}
    plt.figure(figsize=(7, 4)); eda["scale_strategy"].value_counts().plot(kind="bar", color="steelblue")
    plt.title("Train-only Scaling Strategy Counts"); plt.ylabel("Features"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "scaling_strategy_counts.png", dpi=180); plt.close()
    top_association = eda.sort_values("mutual_information", ascending=False).head(30).iloc[::-1]
    plt.figure(figsize=(9, 7)); plt.barh(top_association["feature"], top_association["mutual_information"], color="seagreen")
    plt.title("Top Train-only Mutual Information Features"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "top_mutual_information.png", dpi=180); plt.close()
    _save_state(state)
    return _write_state_outputs(state, feature_frame, subset_index, config)


def run_feature_selection_stage2(feature_frame, subset_index, config=None):
    """Score only Stage-1 candidates with chronological CV and apply its threshold."""
    ensure_directories(); config = config or SelectionConfig(); state = _load_state(feature_frame, subset_index, 1)
    _, _, X_train, y_train = _inputs(feature_frame, subset_index)
    stage2_top, stage2_all = metric_filter(X_train, y_train, state["stage1"]["feature"].tolist(), config)
    state["stage2_all"] = stage2_all.merge(state["stage1"][["feature", "stage1_score"]], on="feature", how="left", validate="one_to_one")
    state["stage2_top"] = state["stage2_all"].set_index("feature").loc[stage2_top["feature"].tolist()].reset_index()
    state["completed"] = 2; _save_state(state)
    state["stage2_all"].to_csv(TABLES_DIR / "selection_stage2_metrics.csv", index=False)
    return _write_state_outputs(state, feature_frame, subset_index, config)


def run_feature_selection_stage3(feature_frame, subset_index, config=None):
    """Run multicollinearity, embedded filtering and optional RFE from Stage-2 output."""
    ensure_directories(); config = config or SelectionConfig(); state = _load_state(feature_frame, subset_index, 2)
    X_all, _, X_train, y_train = _inputs(feature_frame, subset_index); eda = state["eda"]
    ranks = state["stage2_all"].set_index("feature")[config.stage2_metric]
    state["noncollinear"], removed = prune_multicollinearity(X_train, state["stage2_top"]["feature"].tolist(), ranks, config.multicollinearity_threshold)
    selected, ranking, _, _, details = embedded_filter(X_train, y_train, eda, state["noncollinear"], config)
    state["embedded_selected"], state["embedded_ranking"] = selected, ranking
    state["final_features"], state["rfe_ranking"] = apply_optional_rfe(X_train, y_train, eda, selected, config)
    transformer, groups = build_column_transformer(eda, state["final_features"]); transformer.fit(X_train[state["final_features"]])
    joblib.dump(transformer, MODELS_DIR / "feature_transformer.joblib")
    pd.DataFrame({"feature": state["final_features"]}).to_csv(TABLES_DIR / "selected_features.csv", index=False)
    removed.to_csv(TABLES_DIR / "multicollinearity_removed.csv", index=False); ranking.to_csv(TABLES_DIR / "selection_stage3_embedded.csv", index=False); state["rfe_ranking"].to_csv(TABLES_DIR / "selection_rfe.csv", index=False)
    pd.concat([X_all[state["final_features"]], feature_frame[META_COLUMNS]], axis=1).to_csv(PROCESSED_DIR / "selected_dataset.csv")
    funnel_names = ["Initial", "Quality", "Hourglass", "Metric", "Non-collinear", "Embedded", "Final"]
    funnel_values = [len(X_all.columns), len(state["quality_features"]), len(state["stage1"]), len(state["stage2_top"]), len(state["noncollinear"]), len(selected), len(state["final_features"])]
    plt.figure(figsize=(8, 4)); plt.plot(funnel_names, funnel_values, marker="o"); plt.title("Three-stage Feature Selection Funnel"); plt.ylabel("Feature Count"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(FIGURES_DIR / "feature_selection_funnel.png", dpi=180); plt.close()
    if len(state["final_features"]) > 1:
        import seaborn as sns
        plt.figure(figsize=(10, 8)); sns.heatmap(X_train[state["final_features"]].corr(), cmap="coolwarm", center=0)
        plt.title("Final Feature Correlation (SubsetTrain)"); plt.tight_layout(); plt.savefig(FIGURES_DIR / "final_feature_correlation.png", dpi=180); plt.close()
    state["completed"] = 3; state["embedded_details"] = details; state["transform_groups"] = groups; _save_state(state)
    return _write_state_outputs(state, feature_frame, subset_index, config)


def run_feature_selection(feature_frame, subset_index, config=None):
    """Backward-compatible one-call runner; the UI uses the staged functions."""
    config = config or SelectionConfig()
    run_feature_selection_stage1(feature_frame, subset_index, config)
    run_feature_selection_stage2(feature_frame, subset_index, config)
    summary = run_feature_selection_stage3(feature_frame, subset_index, config)
    dataset = pd.read_csv(PROCESSED_DIR / "selected_dataset.csv", index_col=0, parse_dates=True)
    return dataset, joblib.load(MODELS_DIR / "feature_transformer.joblib"), summary

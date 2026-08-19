"""Streamlit interface for feature engineering and model evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import load_market_and_macro_data
from feature_pipeline import create_feature_universe
from feature_selection import (
    METRIC_NAMES,
    SELECTION_STATE_PATH,
    SelectionConfig,
    run_feature_selection_stage1,
    run_feature_selection_stage2,
    run_feature_selection_stage3,
)
from model_pipeline import (
    ModelConfig,
    optimize_blending_development,
    train_blending_ensemble,
)
from nvconfig import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR, TABLES_DIR, TARGET_THRESHOLD
from split_utils import chronological_blending_split
from wavelet_pipeline import (
    SUPPORTED_THRESHOLD_RULES,
    SUPPORTED_WAVELETS,
    WaveletConfig,
    denoise_market_data,
)


st.set_page_config(
    page_title="QQQ Blending Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_CSS = """
<style>
    /* This application has no sidebar. */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(37, 99, 235, .08), transparent 25rem),
            radial-gradient(circle at 92% 5%, rgba(14, 165, 233, .07), transparent 24rem),
            #f7f9fc;
    }
    .block-container {
        max-width: 1480px;
        padding-top: 1.35rem;
        padding-bottom: 3rem;
    }
    .app-hero {
        padding: 1.15rem 1.35rem;
        border: 1px solid #dbe4f0;
        border-radius: 18px;
        background: linear-gradient(115deg, #ffffff 0%, #f4f8ff 70%, #edf7ff 100%);
        box-shadow: 0 8px 28px rgba(15, 23, 42, .06);
        margin-bottom: .85rem;
    }
    .app-kicker {
        color: #2563eb;
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .25rem;
    }
    .app-title {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 760;
        line-height: 1.15;
        margin: 0;
    }
    .app-subtitle {
        color: #64748b;
        margin-top: .42rem;
        margin-bottom: 0;
    }
    .page-heading {
        margin: 1.25rem 0 1rem;
    }
    .page-heading h2 {
        color: #0f172a;
        font-size: 1.55rem;
        margin: 0 0 .3rem;
    }
    .page-heading p {
        color: #64748b;
        margin: 0;
    }
    .section-label {
        color: #334155;
        font-size: .82rem;
        font-weight: 750;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin: .2rem 0 .5rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dfe7f1;
        border-radius: 14px;
        background: rgba(255, 255, 255, .86);
        box-shadow: 0 3px 14px rgba(15, 23, 42, .035);
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        background: #ffffff;
        padding: .8rem .9rem;
    }
    div[data-testid="stMetricLabel"] p {
        color: #64748b;
        font-weight: 650;
    }
    div[data-testid="stMetricValue"] {
        color: #0f172a;
    }
    div[data-testid="stSegmentedControl"] > div {
        width: 100%;
        padding: .25rem;
        border: 1px solid #dbe4f0;
        border-radius: 12px;
        background: #ffffff;
    }
    div[data-testid="stSegmentedControl"] label {
        flex: 1;
        justify-content: center;
        min-height: 2.7rem;
    }
    .flow-note {
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        background: #eff6ff;
        color: #334155;
        padding: .75rem 1rem;
        margin: .25rem 0 1rem;
    }
    .footer-note {
        color: #94a3b8;
        font-size: .78rem;
        text-align: center;
        margin-top: 2rem;
    }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 680;
        min-height: 2.65rem;
    }
</style>
"""


def _read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    """Read a CSV artefact if it exists; otherwise return ``None``."""

    return pd.read_csv(path, **kwargs) if path.exists() else None


def _read_json(path: Path) -> dict | None:
    """Read a UTF-8 JSON artefact if it exists; otherwise return ``None``."""

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _parse_numbers(raw: str, cast=float) -> list:
    """Parse a comma-separated hyperparameter list from the web form."""

    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Candidate parameters cannot be empty.")
    return values


FEATURE_UI_SETTINGS_PATH = PROCESSED_DIR / "feature_ui_settings.json"


FEATURE_UI_DEFAULTS = {
    "start": "2019-01-02",
    "target_threshold": TARGET_THRESHOLD,
    "include_credit": True,
    "refresh": False,
    "reuse_feature_cache": True,
    "wavelet_enabled": True,
    "wavelet_name": "db4",
    "wavelet_level": 2,
    "wavelet_threshold_rule": "universal_mad",
    "wavelet_threshold_scale": 1.0,
    "wavelet_threshold_mode": "soft",
    "wavelet_window_size": 128,
    "wavelet_minimum_history": 32,
    "wavelet_field_mode": "prices_and_volume",
    "wavelet_n_jobs": -1,
    "missing_threshold": 0.20,
    "minimum_corr": 0.015,
    "mi_quantile": 0.50,
    "stage1_max": 400,
    "metric": "roc_auc",
    "stage2_min_metric": 0.50,
    "stage2_top": 160,
    "corr_threshold": 0.92,
    "embedded_method": "extra_trees",
    "embedded_top": 60,
    "final_count": 58,
    "rfe_enabled": False,
    "rfe_trigger": 60,
    "selection_n_jobs": -1,
}


def _load_feature_ui_settings() -> dict:
    """Load the last submitted feature configuration for process restarts."""

    saved = _read_json(FEATURE_UI_SETTINGS_PATH) or {}
    return {**FEATURE_UI_DEFAULTS, **saved}


def _save_feature_ui_settings(values: dict) -> None:
    """Persist UI settings; runtime-only cache status is intentionally omitted."""

    FEATURE_UI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_UI_SETTINGS_PATH.write_text(
        json.dumps(values, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _capture_feature_widget_settings() -> None:
    """Save currently edited feature controls before their workspace unmounts."""

    widget_mapping = {
        "start": "fe_start",
        "target_threshold": "fe_target_threshold",
        "include_credit": "fe_include_credit",
        "refresh": "fe_refresh",
        "reuse_feature_cache": "fe_reuse_feature_cache",
        "wavelet_enabled": "fe_wavelet_enabled",
        "wavelet_name": "fe_wavelet_name",
        "wavelet_level": "fe_wavelet_level",
        "wavelet_threshold_rule": "fe_wavelet_threshold_rule",
        "wavelet_threshold_scale": "fe_wavelet_threshold_scale",
        "wavelet_threshold_mode": "fe_wavelet_threshold_mode",
        "wavelet_window_size": "fe_wavelet_window_size",
        "wavelet_minimum_history": "fe_wavelet_minimum_history",
        "wavelet_field_mode": "fe_wavelet_field_mode",
        "wavelet_n_jobs": "fe_wavelet_n_jobs",
        "missing_threshold": "fe_missing_threshold",
        "minimum_corr": "fe_minimum_corr",
        "mi_quantile": "fe_mi_quantile",
        "stage1_max": "fe_stage1_max",
        "metric": "fe_metric",
        "stage2_min_metric": "fe_stage2_min_metric",
        "stage2_top": "fe_stage2_top",
        "corr_threshold": "fe_corr_threshold",
        "embedded_method": "fe_embedded_method",
        "embedded_top": "fe_embedded_top",
        "final_count": "fe_final_count",
        "rfe_enabled": "fe_rfe_enabled",
        "rfe_trigger": "fe_rfe_trigger",
        "selection_n_jobs": "fe_selection_n_jobs",
    }
    current = _load_feature_ui_settings()
    captured = {
        setting: st.session_state.get(widget_key, current[setting])
        for setting, widget_key in widget_mapping.items()
    }
    _save_feature_ui_settings(captured)


MODEL_UI_SETTINGS_PATH = PROCESSED_DIR / "model_ui_settings.json"


MODEL_UI_DEFAULTS = {
    "auto_tune": True,
    "base_iterations": 100,
    "meta_iterations": 60,
    "base_cv_splits": 4,
    "meta_cv_splits": 5,
    "metric": "roc_auc",
    "threshold": 0.17644062823057174,
    "optimize_threshold": True,
    "threshold_objective": "total_return",
    "cost": 10.0,
    "evaluation_mode": "static",
    "walk_forward_frequency": "monthly",
    "lr_c": "0.1",
    "lr_penalty": ["l1"],
    "lr_fit_intercept": [True],
    "lr_tol": "0.0001",
    "lr_class_weight": ["none"],
    "svc_c": "0.3",
    "svc_gamma": "0.00003",
    "svc_kernel": ["rbf"],
    "svc_shrinking": [True],
    "svc_tol": "0.0001",
    "svc_degree": "3",
    "svc_coef0": "0",
    "svc_class_weight": ["balanced"],
    "knn_k": "21",
    "knn_weights": ["distance"],
    "knn_p": [2],
    "knn_leaf_size": "20",
    "knn_algorithm": ["auto"],
    "xgb_estimators": "60",
    "xgb_depth": "1",
    "xgb_lr": "0.08",
    "xgb_subsample": "1.0",
    "xgb_colsample": "1.0",
    "xgb_min_child_weight": "10",
    "xgb_gamma": "0.1",
    "xgb_reg_alpha": "0",
    "xgb_reg_lambda": "1",
    "xgb_grow_policy": ["depthwise"],
    "xgb_max_leaves": "0",
    "xgb_max_bin": "128",
    "xgb_class_weight": "balanced",
    "xgb_monotone": True,
    "auto_feature_counts": "58",
}


def _load_model_ui_settings() -> dict:
    """Load the most recently saved model form across Streamlit restarts."""

    saved = _read_json(MODEL_UI_SETTINGS_PATH) or {}
    return {**MODEL_UI_DEFAULTS, **saved}


def _save_model_ui_settings(values: dict) -> None:
    """Persist every model, CV, evaluation and backtest control."""

    MODEL_UI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_UI_SETTINGS_PATH.write_text(
        json.dumps(values, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _saved_index(options, value, fallback=0):
    """Return a safe selectbox index when an older settings file is loaded."""

    return options.index(value) if value in options else fallback


def _hero() -> None:
    """Render the shared project identity block."""

    st.markdown(
        """
        <div class="app-hero">
            <div class="app-kicker">Machine Learning · Quant Research</div>
            <div class="app-title">QQQ Blending Ensemble Lab</div>
            <p class="app-subtitle">
                From 500+ candidate features to LR / SVC / KNN base learners and an XGBoost meta-model,
                the workflow preserves chronological order and isolates Final Validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _page_heading(title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-heading"><h2>{title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def _image_grid(image_names: list[str]) -> None:
    """Show existing chart artefacts in a responsive two-column grid."""

    for index in range(0, len(image_names), 2):
        left, right = st.columns(2, gap="large")
        left_path = FIGURES_DIR / image_names[index]
        if left_path.exists():
            left.image(str(left_path), width="stretch")
        if index + 1 < len(image_names):
            right_path = FIGURES_DIR / image_names[index + 1]
            if right_path.exists():
                right.image(str(right_path), width="stretch")


def _feature_configuration_form() -> tuple[bool, dict]:
    """Render feature-engineering controls and return submitted values."""

    defaults = _load_feature_ui_settings()
    with st.container():
        with st.container(border=True):
            st.markdown('<div class="section-label">01 · Data and Target</div>', unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5, gap="large")
            start = col1.text_input(
                "Start date",
                defaults["start"],
                help="Daily prediction should normally use at least five years of history.",
                key="fe_start",
                persist_state="session",
            )
            target_threshold = col2.number_input(
                "Positive-label threshold",
                min_value=0.0,
                max_value=0.02,
                value=float(defaults["target_threshold"]),
                step=0.0005,
                format="%.4f",
                help="Label an observation as 1 when the next-day return is strictly above this value.",
                key="fe_target_threshold",
                persist_state="session",
            )
            include_credit = col3.toggle(
                "Load credit-spread proxies",
                bool(defaults["include_credit"]),
                help=(
                    "Load FRED market credit-risk proxies, including ICE BofA high-yield, corporate, and BBB option-adjusted spreads. "
                    "These are not single-name CDS quotations."
                ),
                key="fe_include_credit",
                persist_state="session",
            )
            refresh = col4.toggle(
                "Refresh external data",
                bool(defaults["refresh"]),
                help="Ignore market and macro caches and retry external downloads.",
                key="fe_refresh",
                persist_state="session",
            )
            reuse_feature_cache = col5.toggle(
                "Reuse feature cache",
                bool(defaults["reuse_feature_cache"]),
                help="Reuse the 500+ computed predictors when data and wavelet settings are unchanged. Label and selection changes do not rebuild predictors.",
                key="fe_reuse_feature_cache",
                persist_state="session",
            )
            st.caption(
                f"Current target: class 1 when next-day QQQ return > {target_threshold:.2%}; "
                "small gains, zero returns, and negative returns are class 0."
            )

            with st.expander("Credit-spread proxies and data sources", expanded=False):
                st.markdown(
                    """
                    Credit spreads approximate the extra compensation on corporate debt relative to lower-risk government debt.
                    Wider spreads generally indicate higher perceived default or liquidity risk; narrower spreads generally indicate improved risk appetite.
                    The requested series are:

                    - ICE BofA US High Yield OAS;
                    - ICE BofA US Corporate OAS;
                    - ICE BofA BBB US Corporate OAS;
                    - Chicago Fed NFCI, the 10Y-2Y term spread, 2Y/10Y Treasury yields, and the federal funds rate.

                    The data are requested from FRED or the DBnomics FRED mirror and are used only as credit and macro state proxies.
                    **They are not single-name CDS spreads.** Failed downloads are recorded in provenance; no synthetic values are created.
                    """
                )

        with st.container(border=True):
            st.markdown('<div class="section-label">02 · Optional Wavelet Denoising</div>', unsafe_allow_html=True)
            w1, w2, w3, w4 = st.columns(4, gap="large")
            wavelet_enabled = w1.toggle(
                "Enable causal wavelet denoising",
                bool(defaults["wavelet_enabled"]),
                help="Each date uses a trailing window ending on that date; future observations are excluded.",
                key="fe_wavelet_enabled",
                persist_state="session",
            )
            wavelet_name = w2.selectbox(
                "Wavelet basis",
                SUPPORTED_WAVELETS,
                index=list(SUPPORTED_WAVELETS).index(defaults["wavelet_name"]),
                key="fe_wavelet_name",
                persist_state="session",
            )
            wavelet_level = w3.slider(
                "Decomposition level",
                1,
                4,
                int(defaults["wavelet_level"]),
                key="fe_wavelet_level",
                persist_state="session",
            )
            wavelet_threshold_rule = w4.selectbox(
                "Threshold rule",
                SUPPORTED_THRESHOLD_RULES,
                index=list(SUPPORTED_THRESHOLD_RULES).index(defaults["wavelet_threshold_rule"]),
                format_func=lambda value: {
                    "universal_mad": "Universal-MAD (recommended)",
                    "paper_std": "Paper-STD (paper mode)",
                }[value],
                help="Paper-STD uses each detail level's coefficient standard deviation, following page 11 of the paper. Universal-MAD is more robust to outliers.",
                key="fe_wavelet_threshold_rule",
                persist_state="session",
            )
            w5, w6, w7, w8 = st.columns(4, gap="large")
            wavelet_threshold_scale = w5.slider(
                "Threshold scale",
                0.0,
                2.0,
                float(defaults["wavelet_threshold_scale"]),
                0.05,
                key="fe_wavelet_threshold_scale",
                persist_state="session",
            )
            wavelet_threshold_mode = w6.selectbox(
                "Threshold mode",
                ["soft", "hard"],
                index=["soft", "hard"].index(defaults["wavelet_threshold_mode"]),
                format_func=lambda value: {"soft": "Soft (recommended)", "hard": "Hard"}[value],
                key="fe_wavelet_threshold_mode",
                persist_state="session",
            )
            window_options = [32, 64, 128, 256]
            wavelet_window_size = w7.select_slider(
                "Causal rolling window",
                options=window_options,
                value=int(defaults["wavelet_window_size"]),
                key="fe_wavelet_window_size",
                persist_state="session",
            )
            history_options = [16, 24, 32, 48, 64]
            wavelet_minimum_history = w8.select_slider(
                "Minimum history",
                options=history_options,
                value=int(defaults["wavelet_minimum_history"]),
                key="fe_wavelet_minimum_history",
                persist_state="session",
            )
            w9, w10 = st.columns(2, gap="large")
            wavelet_field_mode = w9.selectbox(
                "Denoised fields",
                ["prices", "prices_and_volume", "prices_volume_and_macro"],
                index=["prices", "prices_and_volume", "prices_volume_and_macro"].index(defaults["wavelet_field_mode"]),
                format_func=lambda value: {
                    "prices": "OHLC prices",
                    "prices_and_volume": "OHLC prices and volume",
                    "prices_volume_and_macro": "OHLC, volume, and macro/credit (experimental)",
                }[value],
                key="fe_wavelet_field_mode",
                persist_state="session",
            )
            wavelet_n_jobs = w10.selectbox(
                "Wavelet workers",
                [-1, 1, 2, 4, 8],
                index=[-1, 1, 2, 4, 8].index(int(defaults["wavelet_n_jobs"])),
                format_func=lambda value: "Automatic (up to 8 cores)" if value == -1 else f"{value} processes",
                help="Fields can be denoised in parallel. Select 1 for small data or debugging.",
                key="fe_wavelet_n_jobs",
                persist_state="session",
            )
            st.caption(
                "Denoised market series are used only for predictor construction. target, realized_return, and backtest returns always use raw QQQ closes."
            )
            with st.expander("Paper method and implementation mapping", expanded=False):
                st.markdown(
                    """
                    Page 11 presents a continuous wavelet-transform formula and describes coefficient screening by standard deviation before reconstruction,
                    but it does not report the mother wavelet, decomposition scales, threshold direction, or soft/hard rule.

                    - **Paper-STD:** computes a separate standard-deviation threshold for each DWT detail level and shrinks small high-frequency coefficients.
                    - **Universal-MAD:** estimates noise from the finest detail level with MAD and applies the universal threshold; it is more robust to jumps and outliers.
                    - A literal rule that removes coefficients above their standard deviation would preferentially remove large moves, so it is not implemented as denoising.
                    - Both modes use a trailing window ending on each date to prevent look-ahead bias.
                    """
                )

        with st.container(border=True):
            st.markdown('<div class="section-label">03 · Three-Stage Feature Selection</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4, gap="large")
            missing_threshold = c1.slider(
                "Maximum missing ratio",
                0.0,
                0.8,
                float(defaults["missing_threshold"]),
                0.05,
                help="Computed on SubsetTrain only. A value of 0.20 removes features with more than 20% missing values.",
                key="fe_missing_threshold",
                persist_state="session",
            )
            minimum_corr = c2.slider(
                "Hourglass minimum absolute correlation",
                0.0,
                0.20,
                float(defaults["minimum_corr"]),
                0.005,
                help="Minimum absolute Pearson correlation with the binary target. A feature may pass either the correlation or MI condition.",
                key="fe_minimum_corr",
                persist_state="session",
            )
            mi_quantile = c3.slider(
                "Hourglass MI quantile",
                0.0,
                0.95,
                float(defaults["mi_quantile"]),
                0.05,
                help="Quantile threshold among positive MI values. A value of 0.50 requires MI at or above the median positive MI.",
                key="fe_mi_quantile",
                persist_state="session",
            )
            stage1_max = c4.slider(
                "Hourglass maximum features",
                100,
                800,
                int(defaults["stage1_max"]),
                25,
                help="Maximum number of features entering Stage 2 after ranking those that pass the correlation or MI threshold.",
                key="fe_stage1_max",
                persist_state="session",
            )

            c5, c6, c7, c8, c9 = st.columns(5, gap="large")
            metric = c5.selectbox(
                "Stage 2 evaluation metric",
                METRIC_NAMES,
                index=list(METRIC_NAMES).index(defaults["metric"]),
                key="fe_metric",
                persist_state="session",
            )
            stage2_min_metric = c6.slider(
                "Stage 2 minimum score (strict >)",
                0.0,
                0.90,
                float(defaults["stage2_min_metric"]),
                0.01,
                help="Set to 0 for ranking only. For example, 0.55 retains only features with cross-validated ROC-AUC > 0.55.",
                key="fe_stage2_min_metric",
                persist_state="session",
            )
            stage2_top = c7.slider(
                "Metric-filter feature count", 40, 400, int(defaults["stage2_top"]), 10,
                key="fe_stage2_top", persist_state="session",
            )
            corr_threshold = c8.slider(
                "Collinearity correlation threshold", 0.70, 0.99, float(defaults["corr_threshold"]), 0.01,
                key="fe_corr_threshold", persist_state="session",
            )
            embedded_method = c9.selectbox(
                "Embedded method",
                ["extra_trees", "l1_logistic", "lasso"],
                index=["extra_trees", "l1_logistic", "lasso"].index(defaults["embedded_method"]),
                format_func=lambda value: {
                    "extra_trees": "Extra Trees",
                    "l1_logistic": "L1 Logistic",
                    "lasso": "LASSO (LassoCV)",
                }[value],
                help="LASSO uses the binary target for sparse linear screening; alpha is selected by TimeSeriesSplit inside SubsetTrain only.",
                key="fe_embedded_method",
                persist_state="session",
            )

            with st.expander("How to interpret hourglass parameters", expanded=False):
                st.markdown(
                    """
                    - **Maximum missing ratio 0.20:** retains features with no more than 20% missing values in SubsetTrain.
                    - **Minimum absolute correlation 0.08:** requires
                      $|\\mathrm{Corr}(X_j,y)|\\ge 0.08$, although a nonlinear feature may still pass the MI condition.
                    - **MI quantile 0.50:** sets the MI threshold to the median of positive MI values.
                    - **Maximum features 400:** keeps at most the top 400 combined-rank features after the correlation-or-MI condition.

                    Values printed below a slider are its minimum and maximum range; the highlighted value is the active setting.
                    """
                )

            with st.expander("Advanced selection settings", expanded=False):
                c9, c10, c11, c12, c13 = st.columns(5, gap="large")
                embedded_top = c9.slider(
                    "Embedded feature count", 20, 200, int(defaults["embedded_top"]), 5,
                    key="fe_embedded_top", persist_state="session",
                )
                final_count = c10.slider(
                    "Final feature count", 10, 100, int(defaults["final_count"]), 5,
                    key="fe_final_count", persist_state="session",
                )
                rfe_enabled = c11.toggle(
                    "Enable RFE for high dimension", bool(defaults["rfe_enabled"]),
                    key="fe_rfe_enabled", persist_state="session",
                )
                rfe_trigger = c12.slider(
                    "RFE trigger dimension", 30, 150, int(defaults["rfe_trigger"]), 5,
                    key="fe_rfe_trigger", persist_state="session",
                )
                selection_n_jobs = c13.selectbox(
                    "Selection workers",
                    [-1, 1, 2, 4, 8],
                    index=[-1, 1, 2, 4, 8].index(int(defaults["selection_n_jobs"])),
                    format_func=lambda value: "Automatic (up to 8 cores)" if value == -1 else f"{value} workers",
                    help="Parallelizes MI, univariate time-series scoring, and embedded tree/LASSO calculations.",
                    key="fe_selection_n_jobs",
                    persist_state="session",
                )

        st.markdown('<div class="section-label">04 · Staged Execution</div>', unsafe_allow_html=True)
        st.caption("Each stage saves its outputs. After changing a setting, rerun that stage and all later stages.")
        b1, b2, b3, b4, b5 = st.columns(5, gap="small")
        action = None
        if b1.button("1. Data and wavelet", type="primary", width="stretch", key="prepare_wavelet"):
            action = "wavelet"
        elif b2.button("2. Build features", width="stretch", key="build_universe"):
            action = "universe"
        elif b3.button("3. Stage 1 · Hourglass", width="stretch", key="run_stage1"):
            action = "stage1"
        elif b4.button("4. Stage 2 · Metrics", width="stretch", key="run_stage2"):
            action = "stage2"
        elif b5.button("5. Stage 3 · Final selection", width="stretch", key="run_stage3"):
            action = "stage3"

    values = {
        "start": start,
        "target_threshold": target_threshold,
        "include_credit": include_credit,
        "refresh": refresh,
        "reuse_feature_cache": reuse_feature_cache,
        "wavelet_enabled": wavelet_enabled,
        "wavelet_name": wavelet_name,
        "wavelet_level": wavelet_level,
        "wavelet_threshold_rule": wavelet_threshold_rule,
        "wavelet_threshold_scale": wavelet_threshold_scale,
        "wavelet_threshold_mode": wavelet_threshold_mode,
        "wavelet_window_size": wavelet_window_size,
        "wavelet_minimum_history": wavelet_minimum_history,
        "wavelet_field_mode": wavelet_field_mode,
        "wavelet_n_jobs": wavelet_n_jobs,
        "missing_threshold": missing_threshold,
        "minimum_corr": minimum_corr,
        "mi_quantile": mi_quantile,
        "stage1_max": stage1_max,
        "metric": metric,
        "stage2_min_metric": stage2_min_metric,
        "stage2_top": stage2_top,
        "corr_threshold": corr_threshold,
        "embedded_method": embedded_method,
        "embedded_top": embedded_top,
        "final_count": final_count,
        "rfe_enabled": rfe_enabled,
        "rfe_trigger": rfe_trigger,
        "selection_n_jobs": selection_n_jobs,
    }
    return action, values


def _selection_config(values: dict) -> SelectionConfig:
    return SelectionConfig(
        missing_threshold=values["missing_threshold"], minimum_abs_correlation=values["minimum_corr"],
        minimum_mi_quantile=values["mi_quantile"], stage1_max_features=values["stage1_max"],
        stage2_metric=values["metric"], stage2_min_metric=values["stage2_min_metric"],
        stage2_top_features=values["stage2_top"], multicollinearity_threshold=values["corr_threshold"],
        embedded_method=values["embedded_method"], embedded_top_features=values["embedded_top"],
        rfe_enabled=values["rfe_enabled"], rfe_trigger=values["rfe_trigger"],
        final_feature_count=values["final_count"], n_jobs=values["selection_n_jobs"],
    )


def _load_feature_universe() -> pd.DataFrame:
    universe_path = PROCESSED_DIR / "feature_universe.csv"
    if not universe_path.exists():
        raise ValueError("The feature universe has not been generated. Complete Step 2 first.")
    return pd.read_csv(universe_path, index_col=0, parse_dates=True)


def _invalidate_selection_results() -> None:
    """Prevent stale final features from being shown or used after upstream changes."""
    pd.Series(
        {"completed_stage": 0, "final_dataset_ready": False, "message": "Upstream data or factor inputs changed; rerun staged selection."},
        dtype=object,
    ).to_json(TABLES_DIR / "feature_selection_summary.json", indent=2)


def _run_feature_engineering(action: str, values: dict) -> None:
    """Execute exactly one persistent feature-engineering workflow step."""

    if values["wavelet_minimum_history"] > values["wavelet_window_size"]:
        raise ValueError("Minimum wavelet history cannot exceed the causal rolling window.")
    _save_feature_ui_settings(values)
    if action == "wavelet":
        with st.status("Preparing data and wavelet denoising...", expanded=True) as status:
            st.write("Loading market, cross-asset, and macro data")
            data, provenance = load_market_and_macro_data(
                start=values["start"], include_credit_spreads=values["include_credit"], refresh=values["refresh"],
            )
            feature_source = data
            wavelet_summary = {"enabled": False, "target_and_backtest_use_raw_qqq_prices": True}
            if values["wavelet_enabled"]:
                st.write("Applying wavelet denoising with causal rolling windows")
                config = WaveletConfig(
                    wavelet=values["wavelet_name"], level=values["wavelet_level"], threshold_rule=values["wavelet_threshold_rule"],
                    threshold_scale=values["wavelet_threshold_scale"], threshold_mode=values["wavelet_threshold_mode"],
                    window_size=values["wavelet_window_size"], minimum_history=values["wavelet_minimum_history"],
                    field_mode=values["wavelet_field_mode"], n_jobs=values["wavelet_n_jobs"],
                )
                _, wavelet_summary, _ = denoise_market_data(feature_source, config)
            (TABLES_DIR / "feature_preprocessing_summary.json").write_text(json.dumps({"wavelet": wavelet_summary, "data": provenance, "factor_cache_requested": bool(values["reuse_feature_cache"])}, ensure_ascii=False, indent=2), encoding="utf-8")
            _invalidate_selection_results()
            status.update(label="Step 1 complete: data and wavelet outputs saved", state="complete", expanded=False)
        st.success("Next: select '2. Build features'.")
        return

    if action == "universe":
        raw_path = RAW_DIR / "aligned_market_macro.csv"
        if not raw_path.exists():
            raise ValueError("Market data are not ready. Complete Step 1 first.")
        with st.status("Building or loading the feature cache...", expanded=True) as status:
            raw_data = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            prep = _read_json(TABLES_DIR / "feature_preprocessing_summary.json") or {}
            wavelet_info = prep.get("wavelet", {})
            source = pd.read_pickle(PROCESSED_DIR / "wavelet_denoised_market.pkl") if wavelet_info.get("enabled") else raw_data
            _, cache_info = create_feature_universe(source, target_threshold=values["target_threshold"], target_data=raw_data, use_cache=values["reuse_feature_cache"], return_cache_info=True)
            _invalidate_selection_results()
            status.update(label="Step 2 complete: feature universe saved", state="complete", expanded=False)
        st.success(f"Generated {cache_info['predictor_count']:,} predictors. Next: run Stage 1.")
        return

    universe = _load_feature_universe()
    subset_index = chronological_blending_split(universe)["subset_train"].index
    config = _selection_config(values)
    runners = {
        "stage1": (run_feature_selection_stage1, "Stage 1 complete: EDA and hourglass outputs saved"),
        "stage2": (run_feature_selection_stage2, "Stage 2 complete: temporal metric scores saved"),
        "stage3": (run_feature_selection_stage3, "Stage 3 complete: final training data saved"),
    }
    if action not in runners:
        return
    runner, label = runners[action]
    with st.status(f"Running {label}...", expanded=True) as status:
        summary = runner(universe, subset_index, config)
        status.update(label=label, state="complete", expanded=False)
    st.success("Review this stage's feature list before continuing." if action != "stage3" else "The final feature set is ready for model training and prediction.")
    with st.expander("View stage summary"):
        st.json(summary)


def _wavelet_audit() -> bool:
    """Render saved wavelet evidence independently of feature-selection state."""
    preprocessing_info = _read_json(TABLES_DIR / "feature_preprocessing_summary.json") or {}
    wavelet_info = preprocessing_info.get("wavelet", {})
    if wavelet_info and wavelet_info.get("enabled"):
        with st.expander("Step 1 · Before and after wavelet denoising", expanded=True):
            st.json(wavelet_info)
            raw_path = RAW_DIR / "aligned_market_macro.csv"
            denoised_path = PROCESSED_DIR / "wavelet_denoised_market.pkl"
            if raw_path.exists() and denoised_path.exists():
                raw_frame = pd.read_csv(raw_path, index_col=0, parse_dates=True)
                denoised_frame = pd.read_pickle(denoised_path)
                available_columns = [
                    column
                    for column in wavelet_info.get("denoised_columns", [])
                    if column in raw_frame and column in denoised_frame
                ]
                if available_columns:
                    default_column = "qqq_close" if "qqq_close" in available_columns else available_columns[0]
                    plot_left, plot_right = st.columns([2, 1], gap="large")
                    plot_column = plot_left.selectbox(
                        "Select denoised field",
                        available_columns,
                        index=available_columns.index(default_column),
                        key="wavelet_visual_field",
                    )
                    display_window = plot_right.selectbox(
                        "Display period",
                        ["Last 60 days", "Last 126 days", "Last 252 days", "Full history"],
                        key="wavelet_visual_window",
                    )
                    periods = {"Last 60 days": 60, "Last 126 days": 126, "Last 252 days": 252}.get(display_window)
                    comparison = pd.DataFrame(
                        {"Raw": raw_frame[plot_column], "Denoised": denoised_frame[plot_column]}
                    ).dropna(how="all")
                    if periods is not None:
                        comparison = comparison.tail(periods)
                    figure = go.Figure()
                    figure.add_trace(go.Scatter(x=comparison.index, y=comparison["Raw"], name="Raw", mode="lines", line={"color": "#8795a1", "width": 1.2}))
                    figure.add_trace(go.Scatter(x=comparison.index, y=comparison["Denoised"], name="Denoised", mode="lines", line={"color": "#0f766e", "width": 2.0}))
                    figure.update_layout(
                        title=f"{plot_column}: before and after denoising",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        height=430,
                        margin={"l": 20, "r": 20, "t": 55, "b": 20},
                        legend={"orientation": "h", "y": 1.1},
                    )
                    st.plotly_chart(figure, width="stretch")
            preview_path = FIGURES_DIR / "wavelet_denoising_preview.png"
            if preview_path.exists():
                st.image(str(preview_path), width="stretch")
            diagnostics = _read_csv(TABLES_DIR / "wavelet_column_diagnostics.csv")
            if diagnostics is not None:
                st.dataframe(diagnostics, width="stretch", hide_index=True)
        return True
    if preprocessing_info:
        st.info("Step 1 is complete, but wavelet denoising was disabled, so no before-and-after chart is available.")
    return False


def _feature_results(summary: dict) -> None:
    """Render saved feature-selection metrics, tables and charts."""

    st.markdown('<div class="section-label">Feature Selection Results</div>', unsafe_allow_html=True)
    cache_info = _read_json(PROCESSED_DIR / "feature_cache_last_run.json")
    if cache_info:
        cache_label = "cache hit" if cache_info.get("hit") else "recomputed and cached"
        st.info(
            f"Feature calculation: {cache_label} · {cache_info.get('predictor_count', 0):,} predictors · "
            "Changing only the target threshold or selection settings does not rebuild predictors."
        )
    funnel_metrics = [
        ("Initial features", "initial_features"),
        ("After quality cleaning", "quality_clean_features"),
        ("After hourglass filter", "stage1_features"),
        ("After metric filter", "stage2_features"),
        ("After collinearity pruning", "after_multicollinearity"),
        ("After embedded selection", "embedded_features"),
        ("Final features", "final_features"),
    ]
    for offset in range(0, len(funnel_metrics), 4):
        metric_columns = st.columns(min(4, len(funnel_metrics) - offset), gap="medium")
        for column, (label, key) in zip(metric_columns, funnel_metrics[offset : offset + 4]):
            column.metric(label, f"{summary.get(key, 0):,}")
    if "stage2_min_metric" in summary:
        threshold_text = (
            "no score threshold"
            if float(summary["stage2_min_metric"]) == 0
            else f"{summary.get('stage2_metric', 'metric')} > {float(summary['stage2_min_metric']):.3f}"
        )
        st.caption(
            f"Stage 2 threshold: {threshold_text}; "
            f"eligible candidates: {summary.get('stage2_eligible_features', summary.get('stage2_features', 0)):,}; "
            f"the top {summary.get('stage2_features', 0):,} then enter collinearity pruning."
        )

    overview_tab, stages_tab, lifecycle_tab, eda_tab, charts_tab = st.tabs(
        ["Workflow Overview", "Stage Feature Lists", "Lifecycle Tracking", "Feature EDA", "Diagnostic Charts"]
    )
    with overview_tab:
        st.markdown(
            '<div class="flow-note">'
            'The full data are split chronologically. All EDA statistics, scalers, and selection rules are fitted on SubsetTrain only. '
            'Holdout trains the XGBoost meta-model; Final Validation is reserved for final evaluation and backtesting.'
            '</div>',
            unsafe_allow_html=True,
        )
        summary_frame = pd.DataFrame(
            {
                "Stage": [item[0] for item in funnel_metrics],
                "Retained features": [summary.get(item[1], 0) for item in funnel_metrics],
            }
        )
        st.dataframe(summary_frame, width="stretch", hide_index=True)

    with stages_tab:
        st.caption(
            "Each table contains features retained at the end of that stage. stage_rank is the within-stage rank; "
            "the remaining columns show all EDA, association, classification, and importance metrics available by that stage."
        )
        stage_specs = [
            ("Initial", "00_initial", "Initial display ranking by MI and absolute correlation; no numeric feature has been removed."),
            ("Quality Clean", "01_quality_clean", "Removes features that exceed missingness limits or are constant or zero-variance."),
            ("Hourglass", "02_hourglass", "Applies absolute-correlation or MI thresholds and ranks by the combined hourglass score."),
            ("Metric Filter", "03_metric", "Adds time-series cross-validation metrics and retains features above the Stage 2 threshold."),
            ("Noncollinear", "04_noncollinear", "Greedily removes candidates highly correlated with higher-ranked retained features."),
            ("Embedded", "05_embedded", "Adds embedded importance from Extra Trees, L1 logistic regression, or LASSO."),
            ("RFE / Final", "06_final", "Adds RFE rank and final selection status; these columns enter model training."),
        ]
        stage_tabs = st.tabs([label for label, _, _ in stage_specs])
        for tab, (label, stage_id, explanation) in zip(stage_tabs, stage_specs):
            with tab:
                table = _read_csv(TABLES_DIR / f"feature_stage_{stage_id}.csv")
                if table is None:
                    st.info("This stage table is unavailable. Rerun feature engineering.")
                    continue
                left, right = st.columns([1, 3], gap="large")
                left.metric("Currently retained", f"{len(table):,}")
                right.caption(explanation)
                query = st.text_input(
                    "Search feature names",
                    key=f"feature_search_{stage_id}",
                    placeholder="Example: momentum, volatility, qqq_return",
                )
                if query:
                    table = table.loc[
                        table["feature"].astype(str).str.contains(query, case=False, regex=False)
                    ]
                st.dataframe(table, width="stretch", height=600, hide_index=True)

    with lifecycle_tab:
        st.caption(
            "Each row represents one initial feature. retained_* indicates survival after a stage, rank_* is the stage rank, "
            "and first_removed_stage records the first elimination stage."
        )
        lifecycle = _read_csv(TABLES_DIR / "feature_selection_lifecycle.csv")
        if lifecycle is not None:
            removal_options = ["All", *sorted(lifecycle["first_removed_stage"].dropna().unique().tolist())]
            removal_stage = st.selectbox("Filter by first removal stage", removal_options)
            if removal_stage != "All":
                lifecycle = lifecycle.loc[lifecycle["first_removed_stage"] == removal_stage]
            st.dataframe(lifecycle, width="stretch", height=620, hide_index=True)
        else:
            st.info("The lifecycle table is unavailable. Rerun feature engineering.")

    with eda_tab:
        eda_table = _read_csv(TABLES_DIR / "feature_eda_all.csv")
        if eda_table is not None:
            st.dataframe(eda_table, width="stretch", height=560)
        else:
            st.info("The per-feature EDA table has not been generated.")

    with charts_tab:
        _image_grid(
            [
                "scaling_strategy_counts.png",
                "top_mutual_information.png",
                "feature_selection_funnel.png",
                "final_feature_correlation.png",
            ]
        )


def render_feature_engineering() -> None:
    """Render the feature-engineering workspace."""

    _page_heading(
        "Feature Engineering",
        "Configure data, target, EDA, and three-stage feature selection. All fitted operations are restricted to SubsetTrain.",
    )
    action, values = _feature_configuration_form()
    if action:
        try:
            _run_feature_engineering(action, values)
        except Exception as error:
            st.exception(error)

    summary = _read_json(TABLES_DIR / "feature_selection_summary.json")
    wavelet_ready = _wavelet_audit()
    if summary and summary.get("completed_stage", 3) >= 1:
        st.divider()
        _feature_results(summary)
    elif not wavelet_ready:
        st.info("No selection results are available. Complete data/wavelet processing, feature construction, and all three selection stages in order.")
    else:
        st.info("Wavelet outputs are available. Next, build features and run Stage 1.")


def _model_configuration_form() -> tuple[bool, bool, dict]:
    """Render model controls and return submitted hyperparameters."""

    defaults = _load_model_ui_settings()
    with st.form("model_training_form", border=False):
        with st.container(border=True):
            st.markdown('<div class="section-label">01 · Training and Backtest Settings</div>', unsafe_allow_html=True)
            c1, c2, c3, c4, c5, c6 = st.columns(6, gap="large")
            auto_tune = c1.toggle("Enable automatic tuning", bool(defaults["auto_tune"]))
            base_iterations = c2.slider("Base-model search iterations", 1, 300, int(defaults["base_iterations"]))
            meta_iterations = c3.slider("Meta-model search iterations", 1, 300, int(defaults["meta_iterations"]))
            metric_options = ["roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]
            metric = c4.selectbox(
                "Tuning metric",
                metric_options,
                index=_saved_index(metric_options, defaults["metric"]),
            )
            threshold = c5.slider("Fixed prediction threshold", 0.0, 1.0, float(defaults["threshold"]), 0.001)
            cost = c6.slider("Transaction cost (bps)", 0.0, 50.0, float(defaults["cost"]), 1.0)
            threshold_col, objective_col, base_cv, meta_cv = st.columns(4, gap="large")
            optimize_threshold = threshold_col.toggle(
                "Optimize threshold on development data", bool(defaults["optimize_threshold"]),
                help="Select the threshold using chronological OOF probabilities from Blend Holdout only; Final Validation is not read.",
            )
            objective_options = ["total_return", "sharpe"]
            threshold_objective = objective_col.selectbox(
                "Threshold objective", objective_options,
                index=_saved_index(objective_options, defaults["threshold_objective"]),
                disabled=not optimize_threshold,
            )
            cv_options = [3, 4, 5]
            base_cv_splits = base_cv.selectbox(
                "Base-model time-series CV folds", cv_options,
                index=_saved_index(cv_options, int(defaults["base_cv_splits"]), 1),
            )
            meta_cv_splits = meta_cv.selectbox(
                "XGBoost meta-model time-series CV folds", cv_options,
                index=_saved_index(cv_options, int(defaults["meta_cv_splits"]), 2),
            )
            evaluation_col, frequency_col = st.columns(2, gap="large")
            evaluation_options = ["Static Final Validation", "Walk-forward backtest"]
            saved_evaluation_label = (
                "Walk-forward backtest" if defaults["evaluation_mode"] == "walk_forward"
                else "Static Final Validation"
            )
            evaluation_mode_label = evaluation_col.selectbox(
                "Final evaluation mode",
                evaluation_options,
                index=_saved_index(evaluation_options, saved_evaluation_label),
            )
            frequency_options = ["Monthly", "Quarterly"]
            saved_frequency_label = "Quarterly" if defaults["walk_forward_frequency"] == "quarterly" else "Monthly"
            walk_forward_frequency_label = frequency_col.selectbox(
                "Walk-forward refit frequency",
                frequency_options,
                index=_saved_index(frequency_options, saved_frequency_label),
                help="Used only in walk-forward mode. Hyperparameters remain frozen; models are refitted.",
            )
            st.caption(
                "Static mode performs one 60/20/20 evaluation. Walk-forward mode refits monthly or quarterly inside the original Final interval. "
                "Every window uses labels available before its prediction date and applies a boundary purge."
            )

        with st.container(border=True):
            mode_label = "Automatic candidate search" if auto_tune else "Fixed parameters (first value only)"
            st.markdown(f'<div class="section-label">02 · {mode_label}</div>', unsafe_allow_html=True)
            st.caption("Separate multiple values with commas. Automatic tuning searches combinations; fixed mode uses only the first value in each field.")
            lr_tab, svc_tab, knn_tab, meta_tab = st.tabs(["Logistic Regression", "SVC", "KNN", "XGBoost Meta-Model"])
            with lr_tab:
                c_lr1, c_lr2, c_lr3, c_lr4, c_lr5 = st.columns(5, gap="large")
                lr_c = c_lr1.text_input("LR · C", defaults["lr_c"], help="Inverse regularization strength; smaller C gives stronger shrinkage.")
                lr_penalty = c_lr2.multiselect("LR · penalty", ["l1", "l2"], defaults["lr_penalty"])
                lr_fit_intercept = c_lr3.multiselect("LR · fit_intercept", [True, False], defaults["lr_fit_intercept"])
                lr_tol = c_lr4.text_input("LR · tol", defaults["lr_tol"], help="Solver stopping tolerance.")
                lr_class_weight = c_lr5.multiselect("LR · class_weight", ["none", "balanced"], defaults["lr_class_weight"])
                st.caption("The solver is fixed to liblinear with max_iter=2000. Time-series CV can compare none and balanced class weights.")
            with svc_tab:
                c_s1, c_s2, c_s3 = st.columns(3, gap="large")
                svc_c = c_s1.text_input("SVC · C", defaults["svc_c"])
                svc_gamma = c_s2.text_input("SVC · gamma", defaults["svc_gamma"], help="The search also includes scale.")
                svc_kernel = c_s3.multiselect("SVC · kernel", ["rbf", "linear", "poly", "sigmoid"], defaults["svc_kernel"])
                c_s4, c_s5, c_s6, c_s7, c_s8 = st.columns(5, gap="large")
                svc_shrinking = c_s4.multiselect("SVC · shrinking", [True, False], defaults["svc_shrinking"])
                svc_tol = c_s5.text_input("SVC · tol", defaults["svc_tol"])
                svc_degree = c_s6.text_input("SVC · degree", defaults["svc_degree"], help="Applies only to the polynomial kernel; it remains fixed for RBF.")
                svc_coef0 = c_s7.text_input("SVC · coef0", defaults["svc_coef0"], help="Applies only to polynomial and sigmoid kernels; it remains fixed for RBF.")
                svc_class_weight = c_s8.multiselect("SVC · class_weight", ["none", "balanced"], defaults["svc_class_weight"])
                st.caption("probability=True supplies SVC probabilities to XGBoost using the estimator's built-in probability fit.")
            with knn_tab:
                c_k1, c_k2, c_k3, c_k4, c_k5 = st.columns(5, gap="large")
                knn_k = c_k1.text_input("KNN · n_neighbors", defaults["knn_k"])
                knn_weights = c_k2.multiselect("KNN · weights", ["uniform", "distance"], defaults["knn_weights"])
                knn_p = c_k3.multiselect("KNN · Minkowski p", [1, 2], defaults["knn_p"], help="p=1 is Manhattan distance; p=2 is Euclidean distance.")
                knn_leaf_size = c_k4.text_input("KNN · leaf_size", defaults["knn_leaf_size"], help="Primarily affects search speed and memory use.")
                knn_algorithm = c_k5.multiselect("KNN · algorithm", ["auto", "ball_tree", "kd_tree", "brute"], defaults["knn_algorithm"])
                st.caption("Distance is Minkowski. KNN has no native class_weight; inputs are scaled by a transformer fitted inside each training fold.")
            with meta_tab:
                c6, c7, c8 = st.columns(3, gap="large")
                xgb_estimators = c6.text_input("n_estimators", defaults["xgb_estimators"])
                xgb_depth = c7.text_input("max_depth", defaults["xgb_depth"])
                xgb_lr = c8.text_input("learning_rate", defaults["xgb_lr"])
                c9, c10, c11, c12 = st.columns(4, gap="large")
                xgb_subsample = c9.text_input("subsample", defaults["xgb_subsample"])
                xgb_colsample = c10.text_input("colsample_bytree", defaults["xgb_colsample"])
                xgb_min_child_weight = c11.text_input("min_child_weight", defaults["xgb_min_child_weight"])
                xgb_gamma = c12.text_input("gamma", defaults["xgb_gamma"])
                c13, c14, c15 = st.columns(3, gap="large")
                xgb_reg_alpha = c13.text_input("reg_alpha (L1)", defaults["xgb_reg_alpha"])
                xgb_reg_lambda = c14.text_input("reg_lambda (L2)", defaults["xgb_reg_lambda"])
                xgb_grow_policy = c15.multiselect("grow_policy", ["depthwise", "lossguide"], defaults["xgb_grow_policy"])
                c16, c17, c18, c19 = st.columns(4, gap="large")
                xgb_max_leaves = c16.text_input("max_leaves", defaults["xgb_max_leaves"], help="0 means unlimited; leaf count is not searched by default for this small meta sample.")
                xgb_max_bin = c17.text_input("max_bin", defaults["xgb_max_bin"])
                class_weight_options = ["none", "balanced"]
                xgb_class_weight = c18.selectbox(
                    "XGB · class weight", class_weight_options,
                    index=_saved_index(class_weight_options, defaults["xgb_class_weight"], 1),
                )
                xgb_monotone = c19.toggle("Monotonic probability constraints", bool(defaults["xgb_monotone"]), help="The final positive probability cannot decrease when any base positive probability rises while the others are fixed.")
                st.caption("The selected setup uses balanced weighting and monotonic increases in all three base probabilities.")

        with st.container(border=True):
            st.markdown('<div class="section-label">03 · Automatic Development Optimization (Optional)</div>', unsafe_allow_html=True)
            auto_feature_counts = st.text_input(
                "Candidate feature counts",
                defaults["auto_feature_counts"],
                help=(
                    "Build nested feature subsets from the training-period embedded ranking. Automatic optimization compares only "
                    "SubsetTrain/Holdout time-series CV and does not read Final Validation labels."
                ),
            )
            st.caption(
                "For each candidate feature count, the automatic workflow searches LR, SVC, KNN, and XGBoost. "
                "Candidates are ranked by meta-model time-series CV on Blend Holdout, then by higher mean base-model CV and fewer features."
            )

        save_column, train_column, auto_column = st.columns([1, 2, 2], gap="large")
        save_only = save_column.form_submit_button("Save current parameters", width="stretch")
        submitted = train_column.form_submit_button(
            "Train and evaluate blending model", type="primary", icon="▶️", width="stretch"
        )
        auto_submitted = auto_column.form_submit_button(
            "Optimize on development data, then evaluate", icon="🧭", width="stretch"
        )

    values = {
        "auto_tune": auto_tune,
        "base_iterations": base_iterations,
        "meta_iterations": meta_iterations,
        "base_cv_splits": base_cv_splits,
        "meta_cv_splits": meta_cv_splits,
        "metric": metric,
        "threshold": threshold,
        "optimize_threshold": optimize_threshold,
        "threshold_objective": threshold_objective,
        "cost": cost,
        "evaluation_mode": "walk_forward" if evaluation_mode_label.startswith("Walk-forward") else "static",
        "walk_forward_frequency": "monthly" if walk_forward_frequency_label == "Monthly" else "quarterly",
        "lr_c": lr_c,
        "lr_penalty": lr_penalty,
        "lr_fit_intercept": lr_fit_intercept,
        "lr_tol": lr_tol,
        "lr_class_weight": lr_class_weight,
        "svc_c": svc_c,
        "svc_gamma": svc_gamma,
        "svc_kernel": svc_kernel,
        "svc_shrinking": svc_shrinking,
        "svc_tol": svc_tol,
        "svc_degree": svc_degree,
        "svc_coef0": svc_coef0,
        "svc_class_weight": svc_class_weight,
        "knn_k": knn_k,
        "knn_weights": knn_weights,
        "knn_p": knn_p,
        "knn_leaf_size": knn_leaf_size,
        "knn_algorithm": knn_algorithm,
        "xgb_estimators": xgb_estimators,
        "xgb_depth": xgb_depth,
        "xgb_lr": xgb_lr,
        "xgb_subsample": xgb_subsample,
        "xgb_colsample": xgb_colsample,
        "xgb_min_child_weight": xgb_min_child_weight,
        "xgb_gamma": xgb_gamma,
        "xgb_reg_alpha": xgb_reg_alpha,
        "xgb_reg_lambda": xgb_reg_lambda,
        "xgb_grow_policy": xgb_grow_policy,
        "xgb_max_leaves": xgb_max_leaves,
        "xgb_max_bin": xgb_max_bin,
        "xgb_class_weight": xgb_class_weight,
        "xgb_monotone": xgb_monotone,
        "auto_feature_counts": auto_feature_counts,
    }
    if save_only or submitted or auto_submitted:
        _save_model_ui_settings(values)
    if save_only:
        st.success("Model parameters saved. Streamlit restores them after restart.")
    return submitted, auto_submitted, values


def _build_model_config(values: dict, auto_optimization_details: dict | None = None) -> ModelConfig:
    """Convert persisted UI values into the reproducible modelling contract."""

    return ModelConfig(
        auto_tune=values["auto_tune"],
        base_search_iterations=values["base_iterations"],
        meta_search_iterations=values["meta_iterations"],
        base_cv_splits=values["base_cv_splits"],
        meta_cv_splits=values["meta_cv_splits"],
        search_metric=values["metric"],
        prediction_threshold=values["threshold"],
        optimize_threshold=values["optimize_threshold"],
        threshold_objective=values["threshold_objective"],
        transaction_cost_bps=values["cost"],
        meta_class_weight=values["xgb_class_weight"],
        meta_monotone_probabilities=values["xgb_monotone"],
        evaluation_mode=values["evaluation_mode"],
        walk_forward_frequency=values["walk_forward_frequency"],
        auto_optimization_details=auto_optimization_details,
        lr_grid={
            "model__C": _parse_numbers(values["lr_c"]),
            "model__penalty": values["lr_penalty"] or ["l2"],
            "model__fit_intercept": values["lr_fit_intercept"] or [True],
            "model__tol": _parse_numbers(values["lr_tol"]),
            "model__class_weight": [None if item == "none" else item for item in (values["lr_class_weight"] or ["none"])],
        },
        svc_grid={
            "model__C": _parse_numbers(values["svc_c"]),
            "model__gamma": ["scale"] + _parse_numbers(values["svc_gamma"]),
            "model__kernel": values["svc_kernel"] or ["rbf"],
            "model__shrinking": values["svc_shrinking"] or [True],
            "model__tol": _parse_numbers(values["svc_tol"]),
            "model__degree": _parse_numbers(values["svc_degree"], int),
            "model__coef0": _parse_numbers(values["svc_coef0"]),
            "model__class_weight": [None if item == "none" else item for item in (values["svc_class_weight"] or ["none"])],
        },
        knn_grid={
            "model__n_neighbors": _parse_numbers(values["knn_k"], int),
            "model__weights": values["knn_weights"] or ["uniform"],
            "model__p": values["knn_p"] or [2],
            "model__leaf_size": _parse_numbers(values["knn_leaf_size"], int),
            "model__algorithm": values["knn_algorithm"] or ["auto"],
        },
        meta_grid={
            "n_estimators": _parse_numbers(values["xgb_estimators"], int),
            "max_depth": _parse_numbers(values["xgb_depth"], int),
            "learning_rate": _parse_numbers(values["xgb_lr"]),
            "subsample": _parse_numbers(values["xgb_subsample"]),
            "colsample_bytree": _parse_numbers(values["xgb_colsample"]),
            "min_child_weight": _parse_numbers(values["xgb_min_child_weight"]),
            "gamma": _parse_numbers(values["xgb_gamma"]),
            "reg_alpha": _parse_numbers(values["xgb_reg_alpha"]),
            "reg_lambda": _parse_numbers(values["xgb_reg_lambda"]),
            "grow_policy": values["xgb_grow_policy"] or ["depthwise"],
            "max_leaves": _parse_numbers(values["xgb_max_leaves"], int),
            "max_bin": _parse_numbers(values["xgb_max_bin"], int),
        },
    )


def _run_model_training(
    dataset: pd.DataFrame,
    transformer,
    values: dict,
    auto_optimization_details: dict | None = None,
) -> dict:
    """Train and evaluate the Blending ensemble from submitted settings."""

    config = _build_model_config(values, auto_optimization_details)
    with st.status("Training the blending ensemble...", expanded=True) as status:
        st.write("Optimizing LR, SVC, and KNN with time-series cross-validation")
        st.write("Generating Holdout base probabilities and fitting the XGBoost meta-model")
        if config.evaluation_mode == "walk_forward":
            frequency_text = "monthly" if config.walk_forward_frequency == "monthly" else "quarterly"
            st.write(f"Freezing hyperparameters and refitting strict blending windows {frequency_text} inside Final Validation")
        else:
            st.write("Running one static classification evaluation and strategy backtest on the isolated Final Validation set")
        summary = train_blending_ensemble(dataset, transformer, config)
        status.update(label="Model training and final evaluation complete", state="complete", expanded=False)
    if config.evaluation_mode == "walk_forward":
        st.success(
            "Hyperparameter search did not use Final labels. Each rolling prediction block uses only labels available at that time; "
            "earlier blocks may enter later refits, but future labels remain unavailable."
        )
    else:
        st.success("Final Validation labels were not used for feature selection, hyperparameter search, or meta-model training.")
    if summary.get("run_archive"):
        st.info(f"Run archived at: {summary['run_archive']}")
    with st.expander("View run summary"):
        st.json(summary)
    return summary


def _run_auto_development_optimization(values: dict) -> dict:
    """Run leakage-safe auto research, then evaluate its winner on Final Validation."""

    universe_path = PROCESSED_DIR / "feature_universe.csv"
    if not universe_path.exists() or not SELECTION_STATE_PATH.exists():
        raise ValueError("Automatic optimization requires completed feature-engineering state and feature_universe.csv.")
    state = joblib.load(SELECTION_STATE_PATH)
    if state.get("completed", 0) < 3:
        raise ValueError("Complete Stage 3 feature selection before automatic development optimization.")
    ranking = state.get("embedded_ranking")
    if ranking is None or ranking.empty:
        raise ValueError("The training-period embedded feature ranking is missing. Rerun Stage 3 feature selection.")
    feature_counts = _parse_numbers(values["auto_feature_counts"], int)
    if not feature_counts:
        raise ValueError("Candidate feature counts cannot be empty; for example: 20,30,40,48.")

    optimization_values = {**values, "auto_tune": True}
    universe = pd.read_csv(universe_path, index_col=0, parse_dates=True)
    ranked_features = ranking.sort_values("embedded_importance", ascending=False)["feature"].tolist()
    with st.status("Running automatic development optimization...", expanded=True) as status:
        st.write("Searching candidate feature subsets using only SubsetTrain and Blend Holdout time-series cross-validation.")
        winner_frame, winner_transformer, selection, candidate_ranking = optimize_blending_development(
            universe,
            state["eda"],
            ranked_features,
            _build_model_config(optimization_values),
            feature_counts,
        )
        winner_features = selection["winner"]["feature_names"]
        pd.DataFrame({"feature": winner_features}).to_csv(
            TABLES_DIR / "auto_optimized_selected_features.csv", index=False
        )
        winner_frame.to_csv(PROCESSED_DIR / "auto_optimized_selected_dataset.csv")
        joblib.dump(winner_transformer, MODELS_DIR / "auto_optimized_feature_transformer.joblib")
        st.write(
            f"Best development candidate: {len(winner_features)} features; "
            f"XGBoost Holdout time-series CV {selection['winner']['meta_cv_score']:.3f}."
        )
        status.update(label="Development optimization complete; preparing one final evaluation", state="complete", expanded=False)

    return _run_model_training(
        winner_frame,
        winner_transformer,
        optimization_values,
        auto_optimization_details=selection,
    )


def _model_results(summary: dict) -> None:
    """Render saved classification and backtest results."""

    st.markdown('<div class="section-label">Latest Final Validation Results</div>', unsafe_allow_html=True)
    metrics = summary.get("metrics", {})
    metric_columns = st.columns(6, gap="small")
    metric_names = [
        ("ROC AUC", "roc_auc"),
        ("Accuracy", "accuracy"),
        ("Balanced Acc.", "balanced_accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
    ]
    for column, (label, key) in zip(metric_columns, metric_names):
        column.metric(label, f"{metrics.get(key, 0):.3f}")

    backtest = summary.get("backtest", {})
    pnl_columns = st.columns(5, gap="small")
    pnl_columns[0].metric("Strategy return", f"{100 * backtest.get('strategy_total_return', 0):.2f}%")
    pnl_columns[1].metric("Buy and hold", f"{100 * backtest.get('buy_hold_total_return', 0):.2f}%")
    pnl_columns[2].metric("Sharpe", f"{backtest.get('strategy_sharpe', 0):.2f}")
    pnl_columns[3].metric("Maximum drawdown", f"{100 * backtest.get('strategy_max_drawdown', 0):.2f}%")
    pnl_columns[4].metric("Turnover", f"{backtest.get('turnover', 0):.1f}")

    overview_tab, split_tab, diagnostics_tab, backtest_tab, probability_tab, predictions_tab = st.tabs(
        ["Run Summary", "Base-Model Split Performance", "Classification Diagnostics", "Strategy Backtest", "Model Probabilities", "Prediction Details"]
    )
    with overview_tab:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Sample periods and splits**")
            st.json(summary.get("periods", {}), expanded=True)
            if summary.get("walk_forward", {}).get("enabled"):
                st.markdown("**Walk-forward settings**")
                st.json(summary.get("walk_forward", {}), expanded=True)
        with right:
            st.markdown("**Hyperparameter search results**")
            st.json(summary.get("tuning", {}), expanded=True)
            st.markdown("**Trading-threshold source**")
            st.json(summary.get("threshold_optimization", {}), expanded=False)
        auto_optimization = summary.get("auto_development_optimization")
        if auto_optimization:
            st.markdown("**Automatic development optimization record**")
            st.caption(
                "Candidate ranking uses only SubsetTrain/Holdout time-series CV; Final Validation does not participate in selection."
            )
            st.json(auto_optimization, expanded=False)
            candidate_table = _read_csv(TABLES_DIR / "auto_development_optimization_candidates.csv")
            if candidate_table is not None:
                st.dataframe(candidate_table, width="stretch", hide_index=True)
        if summary.get("run_archive"):
            st.markdown("**Run archive directory**")
            st.code(summary["run_archive"], language=None)
        with st.expander("View time-series CV candidate rankings", expanded=False):
            for model_name in ("lr", "svc", "knn", "xgboost_meta"):
                tuning_table = _read_csv(TABLES_DIR / f"tuning_{model_name}.csv")
                if tuning_table is None:
                    continue
                visible = [
                    column for column in tuning_table.columns
                    if column in {"rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", "params"}
                    or column.startswith("param_")
                ]
                st.markdown(f"**{model_name.upper()} · Top 10**")
                st.dataframe(
                    tuning_table.sort_values("rank_test_score")[visible].head(10),
                    width="stretch",
                    hide_index=True,
                )
        if summary.get("walk_forward", {}).get("enabled"):
            refits = _read_csv(TABLES_DIR / "walk_forward_refits.csv", index_col=0)
            if refits is not None:
                with st.expander("View the per-window walk-forward training audit", expanded=False):
                    st.dataframe(refits, width="stretch", hide_index=False)
    with split_tab:
        split_performance = _read_csv(TABLES_DIR / "base_model_split_performance.csv")
        if split_performance is not None:
            st.caption(
                "SubsetTrain metrics are in-sample after final refit and are usually optimistic. Blend Holdout is unseen by the base models "
                "and supplies XGBoost meta-features. Final Validation is the out-of-sample result."
            )
            st.dataframe(
                split_performance,
                width="stretch",
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(format="%.4f")
                    for column in (
                        "target_positive_rate", "probability_mean", "probability_std",
                        "predicted_positive_rate", "roc_auc", "accuracy",
                        "balanced_accuracy", "precision", "recall", "f1",
                    )
                },
            )
            if summary.get("walk_forward", {}).get("enabled"):
                window_performance = _read_csv(
                    TABLES_DIR / "walk_forward_base_split_performance.csv"
                )
                if window_performance is not None:
                    with st.expander("View base-model training/Holdout performance for each walk-forward window", expanded=False):
                        st.dataframe(
                            window_performance,
                            width="stretch",
                            hide_index=True,
                        )
        else:
            st.info("Retrain the model to generate base-model split performance.")
    with diagnostics_tab:
        _image_grid(
            [
                "roc_curve.png",
                "precision_recall_curve.png",
                "confusion_matrix.png",
                "model_comparison.png",
                "meta_feature_correlation.png",
                "meta_feature_importance.png",
            ]
        )
    with backtest_tab:
        _image_grid(["equity_curve.png", "drawdown.png"])
    predictions = _read_csv(
        TABLES_DIR / "final_validation_predictions.csv",
        index_col=0,
        parse_dates=True,
    )
    with probability_tab:
        if predictions is not None:
            probability_columns = [
                column for column in
                ("lr_probability", "svc_probability", "knn_probability", "probability")
                if column in predictions.columns
            ]
            probability_labels = {
                "lr_probability": "LR probability",
                "svc_probability": "SVC probability",
                "knn_probability": "KNN probability",
                "probability": "XGBoost blending probability",
            }
            st.caption(
                "The first three columns are base-learner positive-class probabilities; "
                "probability is the final positive-class probability produced by the "
                "XGBoost meta-model."
            )
            probability_summary = predictions[probability_columns].agg(
                ["mean", "std", "min", "median", "max"]
            ).T
            probability_summary["above_0.5_ratio"] = (
                predictions[probability_columns] >= 0.5
            ).mean()
            probability_summary.index = [
                probability_labels.get(column, column) for column in probability_summary.index
            ]
            st.dataframe(
                probability_summary.reset_index(names="model"),
                width="stretch",
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(format="%.6f")
                    for column in probability_summary.columns
                },
            )

            curve_tab, distribution_tab, table_tab = st.tabs(
                ["Probability History", "Probability Distribution", "Full Probability Table"]
            )
            with curve_tab:
                figure = go.Figure()
                for column in probability_columns:
                    figure.add_trace(
                        go.Scatter(
                            x=predictions.index,
                            y=predictions[column],
                            name=probability_labels.get(column, column),
                            mode="lines",
                        )
                    )
                figure.add_hline(
                    y=float(summary.get("prediction_threshold", 0.5)),
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="prediction threshold",
                )
                figure.update_layout(
                    height=520,
                    yaxis_title="Positive-class probability",
                    xaxis_title="Date",
                    yaxis_range=[0, 1],
                    legend_orientation="h",
                    legend_y=1.08,
                    margin=dict(l=20, r=20, t=55, b=20),
                )
                st.plotly_chart(figure, width="stretch")
            with distribution_tab:
                distribution = go.Figure()
                for column in probability_columns:
                    distribution.add_trace(
                        go.Box(
                            y=predictions[column],
                            name=probability_labels.get(column, column),
                            boxmean="sd",
                        )
                    )
                distribution.update_layout(
                    height=480,
                    yaxis_title="Positive-class probability",
                    yaxis_range=[0, 1],
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(distribution, width="stretch")
            with table_tab:
                probability_table = predictions[probability_columns].copy()
                st.dataframe(
                    probability_table,
                    width="stretch",
                    height=560,
                    column_config={
                        column: st.column_config.NumberColumn(format="%.10f")
                        for column in probability_columns
                    },
                )
                st.download_button(
                    "Download Full Model Probabilities (CSV)",
                    data=probability_table.to_csv(index=True).encode("utf-8-sig"),
                    file_name="final_validation_model_probabilities.csv",
                    mime="text/csv",
                    width="stretch",
                )
        else:
            st.info("Model probabilities are not available yet. Train the model first.")
    with predictions_tab:
        if predictions is not None:
            ordered_columns = [
                column for column in (
                    "lr_probability", "svc_probability", "knn_probability",
                    "probability", "prediction", "target", "refit_id",
                ) if column in predictions.columns
            ]
            st.caption(
                f"{len(predictions):,} rows. This table and the classification metrics "
                "above use the complete Final Validation set."
            )
            st.dataframe(
                predictions[ordered_columns],
                width="stretch",
                height=560,
                column_config={
                    column: st.column_config.NumberColumn(format="%.10f")
                    for column in ordered_columns if "probability" in column
                },
            )
        else:
            st.info("Final Validation prediction details are not available yet.")


def render_model_training() -> None:
    """Render the model-training and prediction workspace."""

    _page_heading(
        "Model Training and Prediction",
        "Configure or search base-learner and meta-model parameters, then review "
        "classification diagnostics and the trading backtest.",
    )
    feature_summary = _read_json(TABLES_DIR / "feature_selection_summary.json") or {}
    if feature_summary.get("completed_stage", 3) < 3:
        st.warning(
            "Feature selection has not completed Stage 3. Finish correlation pruning, "
            "embedded selection, and RFE before training."
        )
        return
    dataset_path = PROCESSED_DIR / "selected_dataset.csv"
    transformer_path = MODELS_DIR / "feature_transformer.joblib"
    if not dataset_path.exists() or not transformer_path.exists():
        st.warning(
            "No training data are available. Open Feature Engineering and complete "
            "the feature-selection workflow first."
        )
        return

    dataset = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    transformer = joblib.load(transformer_path)
    split = chronological_blending_split(dataset)

    split_columns = st.columns(3, gap="large")
    split_columns[0].metric("SubsetTrain · Base Learners", f"{len(split['subset_train']):,}", "About 60%")
    split_columns[1].metric("Holdout · Meta-Model", f"{len(split['holdout']):,}", "About 20%")
    split_columns[2].metric(
        "Final Validation · Final Evaluation",
        f"{len(split['final_validation']):,}",
        "About 20%",
    )
    st.markdown(
        '<div class="flow-note">SubsetTrain → LR / SVC / KNN → Holdout probabilities → '
        'XGBoost → Final Validation classification and one-day-lagged trading backtest</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Class Imbalance and Tuning Protocol", expanded=False):
        distributions = []
        for name, label in (
            ("subset_train", "SubsetTrain"),
            ("holdout", "Holdout"),
            ("final_validation", "Final Validation"),
        ):
            target = split[name]["target"].astype(int)
            counts = target.value_counts().reindex([0, 1], fill_value=0)
            distributions.append(
                {
                    "Partition": label,
                    "Class 0": int(counts.loc[0]),
                    "Class 1": int(counts.loc[1]),
                    "Positive Rate": f"{counts.loc[1] / len(target):.2%}",
                }
            )
        st.dataframe(
            pd.DataFrame(distributions),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Recommended Tuning Sequence", expanded=False):
        st.markdown(
            """
            1. Search model parameters by ROC-AUC with time-series CV inside
               SubsetTrain or Holdout. Keep the trading threshold fixed, or optimize it
               only from Holdout out-of-fold probabilities.
            2. Prioritize parameters that control model complexity: LR `C/penalty`,
               SVC `C/gamma/kernel`, KNN `n_neighbors/weights/p`, and XGBoost
               `max_depth/min_child_weight/learning_rate/n_estimators/regularization`.
            3. Keep `tol`, `leaf_size`, and `algorithm` at their defaults in the first
               pass because they mainly affect convergence or computational efficiency.
            4. After locating a stable region, narrow the candidate ranges and increase
               the search budget. Evaluate Final Validation once; never select parameters
               by repeatedly inspecting it.

            For the current chronological sample, compute is not the binding constraint;
            **sample size and market-regime drift are**. About 100 trials per base learner
            and 60 trials in a narrowed meta-model space are reasonable ceilings that
            limit multiple-comparison bias on Holdout.
            """
        )
        st.markdown(
            """
            - LR and SVC choose between `class_weight=None` and `balanced` through
              time-series CV inside SubsetTrain.
            - XGBoost defaults to `balanced`: tuning uses the earlier SubsetTrain class
              ratio, while the frozen specification is refitted with the complete Holdout
              ratio. The interface can still disable weighting.
            - XGBoost applies monotone-increasing constraints to all three base-model
              probabilities so that a higher positive-class probability is not interpreted
              as a negative signal.
            - KNN has no native `class_weight`; the workflow instead uses scaled features,
              probability blending, and imbalance-aware evaluation metrics.
            - SMOTE and random resampling are excluded because they disturb financial
              time ordering.

            Automatic tuning uses the configured base-model and meta-model
            `TimeSeriesSplit` objects with `gap=1`. GridSearchCV is used when the complete
            candidate grid fits within the trial budget; otherwise RandomizedSearchCV uses
            a fixed seed. Base learners are tuned only inside SubsetTrain, XGBoost only on
            Holdout meta-features, and Final Validation never participates in tuning.
            """
        )

    submitted, auto_submitted, values = _model_configuration_form()
    if auto_submitted:
        try:
            _run_auto_development_optimization(values)
        except Exception as error:
            st.exception(error)
    elif submitted:
        try:
            _run_model_training(dataset, transformer, values)
        except Exception as error:
            st.exception(error)

    summary = _read_json(TABLES_DIR / "model_summary.json")
    if summary:
        st.divider()
        _model_results(summary)
    else:
        st.info(
            "No model evaluation is available. Configure the parameters, then select "
            "Train and Evaluate Blending Model."
        )


st.markdown(APP_CSS, unsafe_allow_html=True)
_hero()

screen = st.segmented_control(
    "Workspace",
    ["🧪 Feature Engineering", "🤖 Model Training and Prediction"],
    default="🧪 Feature Engineering",
    selection_mode="single",
    key="main_workspace",
    on_change=_capture_feature_widget_settings,
    label_visibility="collapsed",
    width="stretch",
)

if screen == "🤖 Model Training and Prediction":
    render_model_training()
else:
    render_feature_engineering()

st.markdown(
    '<div class="footer-note">QQQ Blending Pipeline · Feature engineering, model training, and evaluation</div>',
    unsafe_allow_html=True,
)


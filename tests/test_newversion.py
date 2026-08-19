"""Regression tests for the blending pipeline."""

import unittest
import tempfile

import numpy as np
import pandas as pd
import feature_selection as selection_module
import model_pipeline as model_module

from feature_pipeline import create_feature_universe
from feature_selection import (
    SelectionConfig,
    _feature_lifecycle,
    _stage_snapshot,
    build_feature_eda,
    embedded_filter,
    metric_filter,
    run_feature_selection_stage1,
    run_feature_selection_stage2,
    run_feature_selection_stage3,
)
from model_pipeline import (
    ModelConfig,
    _walk_forward_period_blocks,
    _xgb_estimator,
    optimize_prediction_threshold,
    run_backtest,
    train_blending_ensemble,
)
from sklearn.preprocessing import StandardScaler
from split_utils import chronological_blending_split
from wavelet_pipeline import WaveletConfig, causal_wavelet_denoise_series, denoise_market_data


class SplitTests(unittest.TestCase):
    def test_nominal_1000_rows_follow_600_200_200_boundaries(self):
        frame = pd.DataFrame({"x": np.arange(1000)}, index=pd.date_range("2020-01-01", periods=1000, freq="B"))
        split = chronological_blending_split(frame, purge_size=1)

        self.assertEqual(split["nominal_counts"], {"subset_train": 600, "holdout": 200, "final_validation": 200})
        self.assertEqual(len(split["subset_train"]), 599)
        self.assertEqual(len(split["holdout"]), 199)
        self.assertEqual(len(split["final_validation"]), 200)
        self.assertLess(split["subset_train"].index.max(), split["holdout"].index.min())
        self.assertLess(split["holdout"].index.max(), split["final_validation"].index.min())


class LabelAndBacktestTests(unittest.TestCase):
    def test_return_at_or_below_quarter_percent_is_class_zero(self):
        index = pd.date_range("2020-01-01", periods=260, freq="B")
        returns = np.resize(np.array([0.0025, 0.0024, -0.01, 0.0030]), len(index) - 1)
        close = np.r_[100.0, 100.0 * np.cumprod(1.0 + returns)]
        frame = pd.DataFrame(index=index)
        for asset, scale in (("qqq", 1.0), ("spy", 0.9), ("tlt", 0.7), ("vix", 0.2)):
            values = close * scale + (15.0 if asset == "vix" else 0.0)
            frame[f"{asset}_open"] = values * 0.999
            frame[f"{asset}_high"] = values * 1.003
            frame[f"{asset}_low"] = values * 0.997
            frame[f"{asset}_close"] = values
            frame[f"{asset}_adj_close"] = values
            frame[f"{asset}_volume"] = 1_000_000 + np.arange(len(index))

        features = create_feature_universe(
            frame, target_threshold=0.0025, use_cache=False, persist_outputs=False
        )
        expected = (features["forward_return"] > 0.0025).astype(int)
        pd.testing.assert_series_equal(features["target"], expected, check_names=False)
        self.assertTrue((features.loc[features["forward_return"] <= 0.0025, "target"] == 0).all())

    def test_predictor_cache_is_reused_when_only_label_threshold_changes(self):
        index = pd.date_range("2020-01-01", periods=260, freq="B")
        close = 100.0 * np.exp(np.cumsum(0.0003 + 0.003 * np.sin(np.arange(len(index)))))
        frame = pd.DataFrame(index=index)
        for asset, scale in (("qqq", 1.0), ("spy", 0.9), ("tlt", 0.7), ("vix", 0.2)):
            values = close * scale + (15.0 if asset == "vix" else 0.0)
            frame[f"{asset}_open"] = values * 0.999
            frame[f"{asset}_high"] = values * 1.003
            frame[f"{asset}_low"] = values * 0.997
            frame[f"{asset}_close"] = values
            frame[f"{asset}_adj_close"] = values
            frame[f"{asset}_volume"] = 1_000_000 + np.arange(len(index))

        with tempfile.TemporaryDirectory() as cache_directory:
            _, first = create_feature_universe(
                frame, target_threshold=0.0025, use_cache=True, return_cache_info=True,
                persist_outputs=False, cache_directory=cache_directory,
            )
            second_features, second = create_feature_universe(
                frame, target_threshold=0.0010, use_cache=True, return_cache_info=True,
                persist_outputs=False, cache_directory=cache_directory,
            )

            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertFalse(first["hit"])
            self.assertTrue(second["hit"])
            expected = (second_features["forward_return"] > 0.0010).astype(int)
            pd.testing.assert_series_equal(second_features["target"], expected, check_names=False)

    def test_backtest_uses_lagged_signal(self):
        index = pd.date_range("2024-01-01", periods=4, freq="B")
        meta = pd.DataFrame({"realized_return": [0.5, 0.02, -0.03, 0.04], "forward_return": [9, 9, 9, 9]}, index=index)
        result = run_backtest(meta, np.array([0.9, 0.1, 0.8, 0.2]), 0.5, 0)
        np.testing.assert_array_equal(result["position"], [0, 1, 0, 1])
        np.testing.assert_allclose(result["strategy_return"], [0, 0.02, 0, 0.04])

    def test_threshold_optimizer_uses_same_row_forward_return(self):
        index = pd.date_range("2024-01-01", periods=6, freq="B")
        meta = pd.DataFrame(
            {"forward_return": [-0.03, -0.02, 0.04, 0.05, -0.01, 0.06]},
            index=index,
        )
        probability = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.95])
        threshold, ranking = optimize_prediction_threshold(
            meta, probability, objective="total_return", grid_size=21, min_exposure=0.05
        )

        self.assertGreater(threshold, 0.3)
        self.assertGreater(ranking.iloc[0]["total_return"], 0.14)
        self.assertEqual(ranking.iloc[0]["exposure"], 0.5)

    def test_meta_model_defaults_to_monotone_balanced_probabilities(self):
        config = ModelConfig()
        model = _xgb_estimator(config)

        self.assertEqual(model.get_params()["monotone_constraints"], (1, 1, 1))
        self.assertEqual(config.meta_class_weight, "balanced")
        self.assertEqual(config.meta_grid["max_depth"], [1, 2])


class WalkForwardTests(unittest.TestCase):
    def test_period_blocks_are_chronological_and_non_overlapping(self):
        frame = pd.DataFrame(
            {"x": np.arange(100)},
            index=pd.date_range("2025-01-02", periods=100, freq="B"),
        )
        blocks = _walk_forward_period_blocks(frame, "monthly")

        self.assertGreater(len(blocks), 1)
        self.assertEqual(sum(len(block) for block in blocks), len(frame))
        for earlier, later in zip(blocks, blocks[1:]):
            self.assertLess(earlier.index.max(), later.index.min())

    def test_walk_forward_predictions_record_past_only_refits(self):
        rng = np.random.default_rng(91)
        rows = 180
        index = pd.date_range("2024-01-02", periods=rows, freq="B")
        frame = pd.DataFrame(
            rng.normal(size=(rows, 5)), index=index,
            columns=[f"feature_{number}" for number in range(5)],
        )
        frame["target"] = np.tile([0, 1], rows // 2)
        frame["forward_return"] = np.where(frame["target"].eq(1), 0.004, -0.002)
        frame["realized_return"] = frame["forward_return"].shift(1).fillna(0.0)
        config = ModelConfig(
            auto_tune=False,
            evaluation_mode="walk_forward",
            walk_forward_frequency="monthly",
            transaction_cost_bps=0.0,
            lr_grid={
                "model__C": [0.1], "model__penalty": ["l2"],
                "model__fit_intercept": [True], "model__tol": [1e-4],
                "model__class_weight": ["balanced"],
            },
            svc_grid={
                "model__C": [0.1], "model__gamma": [0.001], "model__kernel": ["rbf"],
                "model__shrinking": [True], "model__tol": [1e-3],
                "model__class_weight": [None], "model__degree": [3], "model__coef0": [0.0],
            },
            knn_grid={
                "model__n_neighbors": [5], "model__weights": ["uniform"],
                "model__p": [1], "model__leaf_size": [30], "model__algorithm": ["auto"],
            },
            meta_grid={
                "n_estimators": [5], "max_depth": [1], "learning_rate": [0.05],
                "subsample": [1.0], "colsample_bytree": [1.0],
                "min_child_weight": [2.0], "reg_alpha": [0.0], "reg_lambda": [1.0],
                "gamma": [0.0], "grow_policy": ["depthwise"],
                "max_leaves": [0], "max_bin": [64],
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            root = Path(directory)
            original = {
                name: getattr(model_module, name)
                for name in ("OUTPUT_DIR", "PROCESSED_DIR", "TABLES_DIR", "MODELS_DIR", "FIGURES_DIR")
            }
            try:
                model_module.OUTPUT_DIR = root / "outputs"
                model_module.PROCESSED_DIR = root / "processed"
                model_module.TABLES_DIR = root / "tables"
                model_module.MODELS_DIR = root / "models"
                model_module.FIGURES_DIR = root / "figures"
                for path in (
                    model_module.OUTPUT_DIR, model_module.PROCESSED_DIR,
                    model_module.TABLES_DIR, model_module.MODELS_DIR, model_module.FIGURES_DIR,
                ):
                    path.mkdir(parents=True, exist_ok=True)
                summary = train_blending_ensemble(frame, StandardScaler(), config)
                audit = pd.read_csv(model_module.TABLES_DIR / "walk_forward_refits.csv")
                predictions = pd.read_csv(model_module.TABLES_DIR / "final_validation_predictions.csv")

                self.assertTrue(summary["walk_forward"]["enabled"])
                self.assertGreater(summary["walk_forward"]["refit_count"], 1)
                self.assertIn("refit_id", predictions)
                self.assertTrue(
                    (pd.to_datetime(audit["holdout_end"]) < pd.to_datetime(audit["prediction_start"])).all()
                )
                self.assertTrue((model_module.TABLES_DIR / "base_model_split_performance.csv").exists())
                self.assertTrue((model_module.TABLES_DIR / "walk_forward_base_split_performance.csv").exists())
                archived_runs = list((model_module.OUTPUT_DIR / "runs").iterdir())
                self.assertEqual(len(archived_runs), 1)
                self.assertIn("_roc-", archived_runs[0].name)
                self.assertIn("_return-", archived_runs[0].name)
                self.assertTrue((archived_runs[0] / "manifest.json").exists())
                self.assertTrue((archived_runs[0] / "tables" / "model_summary.json").exists())
            finally:
                for name, value in original.items():
                    setattr(model_module, name, value)


class EmbeddedSelectionTests(unittest.TestCase):
    def test_lasso_keeps_nonzero_coefficients_and_uses_time_series_cv(self):
        rng = np.random.default_rng(42)
        rows = 240
        X = pd.DataFrame(
            rng.normal(size=(rows, 10)),
            columns=[f"feature_{index}" for index in range(10)],
        )
        y = ((1.5 * X["feature_0"] - X["feature_1"] + rng.normal(size=rows)) > 0.4).astype(int)
        eda = build_feature_eda(X, y)
        selected, ranking, _, _, details = embedded_filter(
            X,
            y,
            eda,
            list(X.columns),
            SelectionConfig(embedded_method="lasso", embedded_top_features=8, cv_splits=3),
        )

        selected_importance = ranking.set_index("feature").loc[selected, "embedded_importance"]
        self.assertTrue((selected_importance > 0).all())
        self.assertIn("feature_0", selected)
        self.assertEqual(details["method"], "lasso")
        self.assertEqual(details["cv"], "TimeSeriesSplit(n_splits=3, gap=1)")
        self.assertGreater(details["selected_alpha"], 0)

    def test_stage_snapshot_preserves_rank_and_cumulative_metrics(self):
        eda = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "mutual_information": [0.3, 0.2, 0.1],
                "missing_ratio": [0.0, 0.1, 0.0],
            }
        )
        metrics = pd.DataFrame(
            {"feature": ["b", "a"], "roc_auc": [0.61, 0.58], "f1": [0.55, 0.50]}
        )
        snapshot = _stage_snapshot(["b", "a"], eda, "03_metric", metric_scores=metrics)

        self.assertEqual(snapshot["feature"].tolist(), ["b", "a"])
        self.assertEqual(snapshot["stage_rank"].tolist(), [1, 2])
        self.assertIn("mutual_information", snapshot)
        self.assertIn("roc_auc", snapshot)

    def test_feature_lifecycle_records_first_removal_stage(self):
        lifecycle = _feature_lifecycle(
            ["a", "b", "c"],
            [
                ("quality", "01_quality_clean", ["a", "b"]),
                ("hourglass", "02_hourglass", ["a"]),
                ("final", "06_final", ["a"]),
            ],
        ).set_index("feature")

        self.assertEqual(lifecycle.loc["c", "first_removed_stage"], "01_quality_clean")
        self.assertEqual(lifecycle.loc["b", "first_removed_stage"], "02_hourglass")
        self.assertTrue(lifecycle.loc["a", "retained_final"])

    def test_metric_filter_applies_strict_configured_threshold(self):
        index = pd.date_range("2021-01-01", periods=80, freq="B")
        rng = np.random.default_rng(7)
        y = pd.Series(np.tile([0, 1], 40), index=index)
        X = pd.DataFrame({"signal": y + rng.normal(0, 0.05, 80), "noise": rng.normal(size=80)}, index=index)
        selected, scores = metric_filter(
            X,
            y,
            ["signal", "noise"],
            SelectionConfig(stage2_metric="roc_auc", stage2_min_metric=0.55, stage2_top_features=10, cv_splits=3, n_jobs=1),
        )

        self.assertTrue((selected["roc_auc"] > 0.55).all())
        self.assertTrue(scores.loc[scores["feature"] == "signal", "passes_stage2_threshold"].iloc[0])


class WaveletTests(unittest.TestCase):
    def test_causal_wavelet_does_not_change_history_when_future_is_appended(self):
        index = pd.date_range("2022-01-03", periods=160, freq="B")
        values = pd.Series(
            100 + np.linspace(0, 20, len(index)) + np.sin(np.arange(len(index)) * 0.7),
            index=index,
        )
        config = WaveletConfig(window_size=64, minimum_history=16, wavelet="db4", level=2)
        full = causal_wavelet_denoise_series(values, config)
        truncated = causal_wavelet_denoise_series(values.iloc[:100], config)
        np.testing.assert_allclose(full.iloc[:100], truncated, rtol=0.0, atol=1e-12, equal_nan=True)

    def test_paper_std_rule_is_supported_and_audited(self):
        index = pd.date_range("2022-01-03", periods=96, freq="B")
        values = pd.Series(
            100 + np.linspace(0, 8, len(index)) + 0.8 * (-1.0) ** np.arange(len(index)),
            index=index,
            name="qqq_close",
        )
        config = WaveletConfig(
            threshold_rule="paper_std",
            threshold_mode="soft",
            window_size=32,
            minimum_history=16,
            wavelet="db2",
            level=2,
        )
        filtered = causal_wavelet_denoise_series(values, config)

        self.assertTrue(np.isfinite(filtered.iloc[15:]).all())
        self.assertGreater(float((values - filtered).abs().sum()), 0.0)

    def test_wavelet_predictors_keep_raw_price_target(self):
        index = pd.date_range("2020-01-01", periods=260, freq="B")
        close = 100.0 * np.exp(np.cumsum(0.0003 + 0.004 * np.sin(np.arange(len(index)))))
        raw = pd.DataFrame(index=index)
        for asset, scale in (("qqq", 1.0), ("spy", 0.9), ("tlt", 0.7), ("vix", 0.2)):
            values = close * scale + (15.0 if asset == "vix" else 0.0)
            raw[f"{asset}_open"] = values * 0.999
            raw[f"{asset}_high"] = values * 1.003
            raw[f"{asset}_low"] = values * 0.997
            raw[f"{asset}_close"] = values
            raw[f"{asset}_adj_close"] = values
            raw[f"{asset}_volume"] = 1_000_000 + np.arange(len(index))
        denoised, summary, _ = denoise_market_data(
            raw,
            WaveletConfig(window_size=32, minimum_history=16, wavelet="db2", level=1),
            persist_outputs=False,
        )
        features = create_feature_universe(
            denoised,
            target_data=raw,
            target_threshold=0.0025,
            use_cache=False,
            persist_outputs=False,
        )
        expected_return = raw["qqq_close"].shift(-1) / raw["qqq_close"] - 1.0
        expected_return = expected_return.reindex(features.index)

        np.testing.assert_allclose(features["forward_return"], expected_return, rtol=0.0, atol=1e-12)
        self.assertTrue(summary["causal"])
        self.assertTrue(summary["target_and_backtest_use_raw_qqq_prices"])

    def test_parallel_macro_mode_handles_signed_series(self):
        index = pd.date_range("2021-01-04", periods=64, freq="B")
        raw = pd.DataFrame(index=index)
        base = 100.0 + np.arange(len(index), dtype=float)
        for field, multiplier in (("open", 0.999), ("high", 1.002), ("low", 0.997), ("close", 1.0), ("adj_close", 1.0)):
            raw[f"qqq_{field}"] = base * multiplier
        raw["qqq_volume"] = 1_000_000 + np.arange(len(index))
        raw["financial_conditions_index"] = np.sin(np.arange(len(index)) / 5.0) - 0.2

        denoised, summary, diagnostics = denoise_market_data(
            raw,
            WaveletConfig(
                field_mode="prices_volume_and_macro",
                n_jobs=2,
                window_size=32,
                minimum_history=16,
                wavelet="db2",
                level=1,
            ),
            persist_outputs=False,
        )

        self.assertEqual(summary["effective_n_jobs"], 2)
        self.assertIn("financial_conditions_index", summary["denoised_columns"])
        self.assertEqual(
            diagnostics.set_index("column").loc["financial_conditions_index", "field_group"],
            "macro_or_credit",
        )
        self.assertTrue((denoised["financial_conditions_index"].iloc[16:] < 0).any())


class StagedSelectionTests(unittest.TestCase):
    def test_three_selection_stages_resume_from_persisted_state(self):
        rng = np.random.default_rng(17)
        index = pd.date_range("2020-01-01", periods=160, freq="B")
        target = pd.Series(np.tile([0, 1], 80), index=index)
        frame = pd.DataFrame(
            {"signal": target + rng.normal(0, 0.03, len(index))}, index=index
        )
        for number in range(7):
            frame[f"noise_{number}"] = rng.normal(size=len(index))
        frame["target"] = target
        frame["forward_return"] = rng.normal(0, 0.01, len(index))
        frame["realized_return"] = rng.normal(0, 0.01, len(index))
        config = SelectionConfig(
            minimum_abs_correlation=0.0,
            minimum_mi_quantile=0.0,
            stage2_min_metric=0.55,
            stage2_top_features=8,
            embedded_top_features=5,
            final_feature_count=3,
            rfe_enabled=False,
            n_jobs=1,
            cv_splits=3,
        )

        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            root = Path(directory)
            original = {
                name: getattr(selection_module, name)
                for name in ("PROCESSED_DIR", "TABLES_DIR", "MODELS_DIR", "FIGURES_DIR", "SELECTION_STATE_PATH")
            }
            try:
                selection_module.PROCESSED_DIR = root / "processed"
                selection_module.TABLES_DIR = root / "tables"
                selection_module.MODELS_DIR = root / "models"
                selection_module.FIGURES_DIR = root / "figures"
                for path in (selection_module.PROCESSED_DIR, selection_module.TABLES_DIR, selection_module.MODELS_DIR, selection_module.FIGURES_DIR):
                    path.mkdir(parents=True, exist_ok=True)
                selection_module.SELECTION_STATE_PATH = selection_module.PROCESSED_DIR / "feature_selection_state.joblib"

                stage1 = run_feature_selection_stage1(frame, index[:96], config)
                self.assertEqual(stage1["completed_stage"], 1)
                self.assertTrue(selection_module.SELECTION_STATE_PATH.exists())
                stage2 = run_feature_selection_stage2(frame, index[:96], config)
                self.assertEqual(stage2["completed_stage"], 2)
                stage3 = run_feature_selection_stage3(frame, index[:96], config)
                self.assertEqual(stage3["completed_stage"], 3)
                self.assertTrue((selection_module.PROCESSED_DIR / "selected_dataset.csv").exists())
            finally:
                for name, value in original.items():
                    setattr(selection_module, name, value)


if __name__ == "__main__":
    unittest.main()

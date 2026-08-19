"""Causal wavelet denoising for chronological market data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from joblib import Parallel, delayed

from nvconfig import FIGURES_DIR, PROCESSED_DIR, TABLES_DIR, ensure_directories


SUPPORTED_WAVELETS = ("db2", "db4", "sym4", "coif1")
SUPPORTED_FIELD_MODES = ("prices", "prices_and_volume", "prices_volume_and_macro")
SUPPORTED_THRESHOLD_RULES = ("universal_mad", "paper_std")


@dataclass(frozen=True)
class WaveletConfig:
    """Parameters for trailing-window discrete-wavelet denoising."""

    wavelet: str = "db4"
    level: int = 2
    threshold_rule: str = "universal_mad"
    threshold_scale: float = 1.0
    threshold_mode: str = "soft"
    window_size: int = 128
    minimum_history: int = 32
    field_mode: str = "prices"
    n_jobs: int = -1

    def validate(self):
        if self.wavelet not in SUPPORTED_WAVELETS:
            raise ValueError(f"Unsupported wavelet: {self.wavelet}")
        if self.level < 1:
            raise ValueError("Wavelet level must be at least one.")
        if self.threshold_rule not in SUPPORTED_THRESHOLD_RULES:
            raise ValueError(f"Unsupported threshold rule: {self.threshold_rule}")
        if self.threshold_scale < 0:
            raise ValueError("Wavelet threshold scale cannot be negative.")
        if self.threshold_mode not in {"soft", "hard"}:
            raise ValueError("Threshold mode must be soft or hard.")
        if self.window_size < 16:
            raise ValueError("Wavelet window must contain at least 16 rows.")
        if not 8 <= self.minimum_history <= self.window_size:
            raise ValueError("Minimum history must be between 8 and the window size.")
        if self.field_mode not in SUPPORTED_FIELD_MODES:
            raise ValueError(f"Unsupported wavelet field mode: {self.field_mode}")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("Wavelet n_jobs must be -1 (automatic) or a positive integer.")
        return self


def _safe_level(wavelet, requested_level, sample_size):
    maximum = pywt.dwt_max_level(sample_size, pywt.Wavelet(wavelet).dec_len)
    return max(0, min(int(requested_level), int(maximum)))


def _denoise_window(values, config):
    # PyWavelets requires a writable C-contiguous buffer.
    values = np.array(values, dtype=float, copy=True, order="C")
    level = _safe_level(config.wavelet, config.level, len(values))
    if level < 1:
        return values.copy()
    coefficients = pywt.wavedec(values, config.wavelet, mode="symmetric", level=level)
    if config.threshold_rule == "paper_std":
        # Apply a separate standard-deviation threshold at each detail level.
        thresholds = [
            config.threshold_scale * float(np.std(part, ddof=1)) if len(part) > 1 else 0.0
            for part in coefficients[1:]
        ]
    else:
        # Universal threshold with robust finest-level noise estimation.
        finest_detail = coefficients[-1]
        sigma = np.median(np.abs(finest_detail - np.median(finest_detail))) / 0.6744897501960817
        universal = config.threshold_scale * sigma * math.sqrt(2.0 * math.log(max(len(values), 2)))
        thresholds = [universal] * (len(coefficients) - 1)

    filtered_details = []
    for part, threshold in zip(coefficients[1:], thresholds):
        if np.isfinite(threshold) and threshold > 0:
            filtered_details.append(pywt.threshold(part, threshold, mode=config.threshold_mode))
        else:
            filtered_details.append(part.copy())
    reconstructed = pywt.waverec(
        [coefficients[0]] + filtered_details,
        config.wavelet,
        mode="symmetric",
    )
    return np.asarray(reconstructed[: len(values)], dtype=float)


def causal_wavelet_denoise_series(series, config):
    """Reconstruct every endpoint from a window ending at that endpoint."""
    config.validate()
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    # Preserve initial gaps rather than backfilling from future observations.
    filled = numeric.ffill()
    result = numeric.copy()
    for endpoint in range(len(numeric)):
        start = max(0, endpoint - config.window_size + 1)
        window = filled.iloc[start : endpoint + 1].dropna()
        if len(window) < config.minimum_history:
            continue
        result.iloc[endpoint] = _denoise_window(window.to_numpy(), config)[-1]
    return result.rename(series.name)


def _selected_columns(frame, field_mode):
    price_suffixes = ("_open", "_high", "_low", "_close", "_adj_close")
    columns = [column for column in frame if column.endswith(price_suffixes)]
    if field_mode in {"prices_and_volume", "prices_volume_and_macro"}:
        columns.extend(column for column in frame if column.endswith("_volume"))
    if field_mode == "prices_volume_and_macro":
        market_columns = set(columns)
        columns.extend(
            column
            for column in frame.select_dtypes(include=["number"]).columns
            if column not in market_columns
        )
    return list(dict.fromkeys(columns))


def _effective_n_jobs(requested_jobs, task_count):
    """Bound automatic parallelism to avoid overwhelming a desktop session."""
    if task_count <= 1:
        return 1
    available = os.cpu_count() or 1
    requested = min(8, available) if requested_jobs == -1 else requested_jobs
    return max(1, min(int(requested), int(task_count)))


def _denoise_column(column, source, config):
    """Denoise one raw input column; safe to execute in parallel."""
    source = pd.to_numeric(source, errors="coerce").astype(float)
    is_volume = column.endswith("_volume")
    is_positive_price = column.endswith(("_open", "_high", "_low", "_close", "_adj_close"))
    # Transform nonnegative market fields; retain signed macro units.
    if is_volume:
        transformed = np.log1p(source.clip(lower=0))
        restore = lambda values: np.expm1(values).clip(lower=0)
    elif is_positive_price:
        transformed = np.log(source.clip(lower=np.finfo(float).tiny))
        restore = np.exp
    else:
        transformed = source
        restore = lambda values: values
    filtered = causal_wavelet_denoise_series(transformed, config)
    restored = pd.Series(restore(filtered), index=source.index, name=column)
    restored = restored.where(source.notna())
    raw_diff_std = float(transformed.diff().std())
    denoised_diff_std = float(filtered.diff().std())
    diagnostic = {
        "column": column,
        "field_group": "volume" if is_volume else "price" if is_positive_price else "macro_or_credit",
        "raw_difference_std": raw_diff_std,
        "denoised_difference_std": denoised_diff_std,
        "roughness_reduction": 1.0 - denoised_diff_std / raw_diff_std if raw_diff_std > 0 else 0.0,
        "raw_denoised_correlation": float(source.corr(restored)),
        "mean_absolute_change": float((source - restored).abs().mean()),
    }
    return column, restored, diagnostic


def _restore_ohlc_constraints(frame):
    repaired = frame.copy()
    assets = sorted({column.rsplit("_", 1)[0] for column in repaired if column.endswith("_close")})
    for asset in assets:
        open_col, high_col, low_col, close_col = (f"{asset}_{field}" for field in ("open", "high", "low", "close"))
        if not all(column in repaired for column in (open_col, high_col, low_col, close_col)):
            continue
        ohlc = repaired[[open_col, high_col, low_col, close_col]]
        repaired[high_col] = ohlc.max(axis=1)
        repaired[low_col] = ohlc.min(axis=1)
    return repaired


def denoise_market_data(market_data, config=None, persist_outputs=True):
    """Denoise price/volume inputs while retaining macro series unchanged."""
    ensure_directories()
    config = (config or WaveletConfig()).validate()
    raw = market_data.copy().sort_index()
    denoised = raw.copy()
    selected_columns = _selected_columns(raw, config.field_mode)
    effective_jobs = _effective_n_jobs(config.n_jobs, len(selected_columns))
    results = Parallel(n_jobs=effective_jobs, prefer="processes")(
        delayed(_denoise_column)(column, raw[column], config) for column in selected_columns
    )
    diagnostic_rows = []
    for column, restored, diagnostic in results:
        denoised[column] = restored
        diagnostic_rows.append(diagnostic)
    denoised = _restore_ohlc_constraints(denoised)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = {
        "enabled": True,
        "causal": True,
        "future_observations_used": False,
        "target_and_backtest_use_raw_qqq_prices": True,
        "selected_config": asdict(config),
        "effective_n_jobs": effective_jobs,
        "threshold_method": (
            "Per-level detail-coefficient standard deviation (paper-inspired; conventional small-coefficient shrinkage)"
            if config.threshold_rule == "paper_std"
            else "Robust MAD noise estimate with Donoho-Johnstone universal threshold"
        ),
        "paper_alignment": {
            "source": "Nti et al. (2020), A comprehensive evaluation of ensemble learning for stock-market prediction, p.11",
            "implemented": "wavelet transform, coefficient thresholding, inverse reconstruction",
            "paper_unspecified": [
                "mother wavelet",
                "decomposition level or CWT scales",
                "threshold direction",
                "hard versus soft thresholding",
                "time-causal fitting protocol",
            ],
            "literal_remove_coefficients_above_std": False,
            "reason": "Large detail coefficients commonly carry jumps or signal; removing them would invert standard wavelet denoising.",
        },
        "denoised_columns": selected_columns,
        "denoised_column_count": len(selected_columns),
    }
    if persist_outputs:
        denoised.to_pickle(PROCESSED_DIR / "wavelet_denoised_market.pkl")
        diagnostics.to_csv(TABLES_DIR / "wavelet_column_diagnostics.csv", index=False)
        (TABLES_DIR / "wavelet_denoising_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        preview = pd.DataFrame(
            {"raw_qqq_close": raw["qqq_close"], "denoised_qqq_close": denoised["qqq_close"]}
        )
        preview.to_csv(TABLES_DIR / "wavelet_qqq_close_preview.csv")
        plot_frame = preview.tail(252)
        plt.figure(figsize=(10, 4.5))
        plt.plot(plot_frame.index, plot_frame["raw_qqq_close"], label="Raw QQQ Close", alpha=0.65)
        plt.plot(plot_frame.index, plot_frame["denoised_qqq_close"], label="Causal Wavelet Close", linewidth=1.4)
        plt.title("Causal Wavelet Denoising Audit")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "wavelet_denoising_preview.png", dpi=180)
        plt.close()
    return denoised, summary, diagnostics

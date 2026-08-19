"""Generate and cache a large library-backed feature universe."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from nvconfig import PROCESSED_DIR, ROOT, TARGET_THRESHOLD, ensure_directories


PRICE_SUFFIXES = ("_open", "_high", "_low", "_close", "_adj_close")
FEATURE_CACHE_VERSION = 2


def _frame_fingerprint(frame):
    """Return a deterministic fingerprint for a chronological input frame."""
    normalized = frame.sort_index()
    hashed_values = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    schema = json.dumps(
        [(str(column), str(dtype)) for column, dtype in normalized.dtypes.items()],
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(schema + hashed_values).hexdigest()


def _predictor_cache_key(data, max_lag, rolling_windows):
    payload = {
        "version": FEATURE_CACHE_VERSION,
        "input_fingerprint": _frame_fingerprint(data),
        "max_lag": int(max_lag),
        "rolling_windows": [int(value) for value in rolling_windows],
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), payload


def _asset_names(data):
    return sorted(
        column[:-6]
        for column in data.columns
        if column.endswith("_close") and not column.endswith("_adj_close")
    )


def _safe_add(store, name, values):
    if name not in store:
        store[name] = pd.Series(values, index=values.index).replace([np.inf, -np.inf], np.nan)


def _technical_features(data, store):
    """Call pandas-ta indicators; no indicator formulas are reimplemented."""
    high, low, close, volume = data["qqq_high"], data["qqq_low"], data["qqq_close"], data["qqq_volume"]
    for length in (5, 7, 10, 14, 20, 30, 42, 60):
        for name, series in (
            (f"ta_rsi_{length}", ta.rsi(close, length=length)),
            (f"ta_roc_{length}", ta.roc(close, length=length)),
            (f"ta_mom_{length}", ta.mom(close, length=length)),
            (f"ta_cci_{length}", ta.cci(high, low, close, length=length)),
            (f"ta_willr_{length}", ta.willr(high, low, close, length=length)),
            (f"ta_atr_{length}", ta.atr(high, low, close, length=length)),
            (f"ta_natr_{length}", ta.natr(high, low, close, length=length)),
            (f"ta_cmf_{length}", ta.cmf(high, low, close, volume, length=length)),
            (f"ta_mfi_{length}", ta.mfi(high, low, close, volume, length=length)),
        ):
            if series is not None:
                _safe_add(store, name, series)

        for prefix, frame in (
            (f"ta_bbands_{length}", ta.bbands(close, length=length)),
            (f"ta_stoch_{length}", ta.stoch(high, low, close, k=length, d=3, smooth_k=3)),
            (f"ta_adx_{length}", ta.adx(high, low, close, length=length)),
        ):
            if frame is not None:
                for idx, column in enumerate(frame.columns):
                    _safe_add(store, f"{prefix}_{idx}_{str(column).lower()}", frame[column])

    for fast, slow, signal in ((5, 20, 5), (8, 24, 9), (12, 26, 9), (20, 50, 10)):
        frame = ta.macd(close, fast=fast, slow=slow, signal=signal)
        if frame is not None:
            for idx, column in enumerate(frame.columns):
                _safe_add(store, f"ta_macd_{fast}_{slow}_{signal}_{idx}", frame[column])


def create_feature_universe(
    data,
    target_threshold=TARGET_THRESHOLD,
    max_lag=20,
    rolling_windows=(5, 10, 20, 40, 60, 120),
    target_data=None,
    use_cache=True,
    return_cache_info=False,
    persist_outputs=True,
    cache_directory=None,
):
    """Create 500+ predictors and attach raw-price labels.

    Predictor calculation is cached independently of ``target_threshold``.
    Consequently, changing the label or any downstream selection setting does
    not recalculate indicators.  ``target_data`` lets denoised predictors keep
    labels, realised returns and backtest returns anchored to raw QQQ prices.
    """
    ensure_directories()
    data = data.copy().sort_index()
    raw_target_data = (target_data if target_data is not None else data).copy().sort_index()
    cache_key, cache_payload = _predictor_cache_key(data, max_lag, rolling_windows)
    cache_dir = Path(cache_directory) if cache_directory is not None else PROCESSED_DIR / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"feature_predictors_{cache_key}.pkl"
    cache_hit = bool(use_cache and cache_path.exists())

    if cache_hit:
        features = pd.read_pickle(cache_path)
    else:
        store = {}
        horizons = sorted(set(range(1, max_lag + 1)) | {21, 42, 63})
        assets = _asset_names(data)

        for asset in assets:
            close_col = f"{asset}_close"
            if close_col not in data:
                continue
            close = pd.to_numeric(data[close_col], errors="coerce")
            log_close = np.log(close.where(close > 0))
            return_1 = close.pct_change()
            for horizon in horizons:
                _safe_add(store, f"{asset}_return_{horizon}", close.pct_change(horizon))
                _safe_add(store, f"{asset}_log_return_{horizon}", log_close.diff(horizon))
            for lag in range(1, max_lag + 1):
                _safe_add(store, f"{asset}_return_1_lag_{lag}", return_1.shift(lag))

            for window in rolling_windows:
                rolling = return_1.rolling(window)
                _safe_add(store, f"{asset}_ret_mean_{window}", rolling.mean())
                _safe_add(store, f"{asset}_volatility_{window}", rolling.std())
                _safe_add(store, f"{asset}_ret_median_{window}", rolling.median())
                _safe_add(store, f"{asset}_ret_min_{window}", rolling.min())
                _safe_add(store, f"{asset}_ret_max_{window}", rolling.max())
                _safe_add(store, f"{asset}_ret_skew_{window}", rolling.skew())
                _safe_add(store, f"{asset}_ret_kurt_{window}", rolling.kurt())
                _safe_add(store, f"{asset}_momentum_{window}", close / close.shift(window) - 1.0)
                mean_price = close.rolling(window).mean()
                std_price = close.rolling(window).std()
                _safe_add(store, f"{asset}_price_z_{window}", (close - mean_price) / std_price)
                _safe_add(store, f"{asset}_drawdown_{window}", close / close.rolling(window).max() - 1.0)
                _safe_add(store, f"{asset}_downside_vol_{window}", return_1.where(return_1 < 0).rolling(window).std())
                _safe_add(store, f"{asset}_upside_vol_{window}", return_1.where(return_1 > 0).rolling(window).std())

            volume_col = f"{asset}_volume"
            if volume_col in data:
                volume = pd.to_numeric(data[volume_col], errors="coerce")
                log_volume = np.log1p(volume.clip(lower=0))
                for horizon in (1, 5, 10, 20):
                    _safe_add(store, f"{asset}_volume_change_{horizon}", volume.pct_change(horizon))
                    _safe_add(store, f"{asset}_log_volume_diff_{horizon}", log_volume.diff(horizon))
                for window in rolling_windows:
                    _safe_add(store, f"{asset}_volume_z_{window}", (log_volume - log_volume.rolling(window).mean()) / log_volume.rolling(window).std())

        qqq_return = pd.to_numeric(data["qqq_close"], errors="coerce").pct_change()
        for asset in assets:
            if asset == "qqq" or f"{asset}_close" not in data:
                continue
            asset_return = data[f"{asset}_close"].pct_change()
            for window in (10, 20, 60, 120):
                covariance = qqq_return.rolling(window).cov(asset_return)
                variance = asset_return.rolling(window).var()
                _safe_add(store, f"qqq_{asset}_corr_{window}", qqq_return.rolling(window).corr(asset_return))
                _safe_add(store, f"qqq_{asset}_beta_{window}", covariance / variance)
                _safe_add(store, f"qqq_{asset}_relative_momentum_{window}", data["qqq_close"].pct_change(window) - data[f"{asset}_close"].pct_change(window))

        market_columns = {column for column in data if column.endswith(PRICE_SUFFIXES) or column.endswith("_volume")}
        macro_columns = [column for column in data.columns if column not in market_columns]
        for column in macro_columns:
            series = pd.to_numeric(data[column], errors="coerce")
            for lag in (1, 2, 5, 10, 20):
                _safe_add(store, f"{column}_lag_{lag}", series.shift(lag))
                _safe_add(store, f"{column}_change_{lag}", series.diff(lag))
            for window in (10, 20, 60, 120):
                _safe_add(store, f"{column}_z_{window}", (series - series.rolling(window).mean()) / series.rolling(window).std())
                _safe_add(store, f"{column}_vol_{window}", series.diff().rolling(window).std())

        _technical_features(data, store)
        calendar = pd.DataFrame(index=data.index)
        calendar["dow_sin"] = np.sin(2 * np.pi * data.index.dayofweek / 7)
        calendar["dow_cos"] = np.cos(2 * np.pi * data.index.dayofweek / 7)
        calendar["month_sin"] = np.sin(2 * np.pi * data.index.month / 12)
        calendar["month_cos"] = np.cos(2 * np.pi * data.index.month / 12)

        features = pd.concat([pd.DataFrame(store, index=data.index), calendar], axis=1)
        features = features.loc[:, ~features.columns.duplicated()].replace([np.inf, -np.inf], np.nan)
        if use_cache:
            features.to_pickle(cache_path)

    raw_close = pd.to_numeric(raw_target_data["qqq_close"], errors="coerce").reindex(features.index)
    forward_return = raw_close.shift(-1) / raw_close - 1.0
    features["realized_return"] = raw_close.pct_change()
    features["forward_return"] = forward_return
    features["target"] = np.where(forward_return.notna(), (forward_return > float(target_threshold)).astype(int), np.nan)
    features = features.loc[features["target"].notna()].copy()
    features["target"] = features["target"].astype(int)
    predictor_count = features.shape[1] - 3
    if predictor_count < 500:
        raise ValueError(f"Initial feature universe has only {features.shape[1] - 3} predictors; at least 500 are required.")
    if persist_outputs:
        features.to_csv(PROCESSED_DIR / "feature_universe.csv")
    try:
        cache_file = cache_path.relative_to(ROOT.parent).as_posix()
    except ValueError:
        cache_file = cache_path.resolve().as_posix()
    cache_info = {
        "enabled": bool(use_cache),
        "hit": cache_hit,
        "cache_key": cache_key,
        "cache_file": cache_file,
        "cache_version": FEATURE_CACHE_VERSION,
        "predictor_count": int(predictor_count),
        "target_threshold_excluded_from_cache_key": True,
        **cache_payload,
    }
    if persist_outputs:
        (PROCESSED_DIR / "feature_cache_last_run.json").write_text(
            json.dumps(cache_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return (features, cache_info) if return_cache_info else features

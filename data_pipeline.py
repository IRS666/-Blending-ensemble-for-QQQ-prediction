"""Download, cache, and chronologically align market and credit-spread data."""

from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import pandas as pd
import requests
import yfinance as yf

from nvconfig import DEFAULT_TICKERS, FRED_SERIES, RAW_DIR, ensure_directories


def _alias(ticker):
    return re.sub(r"[^a-z0-9]+", "_", ticker.lower()).strip("_") or "asset"


def _read_cached(path):
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index() if Path(path).exists() else None


def _download_yfinance(tickers, start, end):
    """Use yfinance's maintained downloader; indicator calculations are not recoded."""
    timezone_cache = RAW_DIR / "yf_cache"
    timezone_cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(timezone_cache))
    downloaded = yf.download(
        tickers=list(tickers), start=start, end=end, auto_adjust=False,
        group_by="ticker", progress=False, threads=True,
    )
    if downloaded.empty:
        raise RuntimeError("yfinance returned no market data.")

    frames = []
    for ticker in tickers:
        if isinstance(downloaded.columns, pd.MultiIndex):
            if ticker not in downloaded.columns.get_level_values(0):
                continue
            frame = downloaded[ticker].copy()
        else:
            frame = downloaded.copy()
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        frame = frame.rename(columns={"adj_close": "adj_close", "adj close": "adj_close"})
        required = ["open", "high", "low", "close"]
        if not set(required).issubset(frame.columns):
            continue
        if "adj_close" not in frame:
            frame["adj_close"] = frame["close"]
        if "volume" not in frame:
            frame["volume"] = 0.0
        alias = _alias(ticker)
        frame = frame[["open", "high", "low", "close", "adj_close", "volume"]]
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame.columns = [f"{alias}_{column}" for column in frame.columns]
        frames.append(frame)
    if not frames:
        raise RuntimeError("No requested ticker produced usable OHLC data.")
    return pd.concat(frames, axis=1).sort_index()


def _download_stooq(tickers, start, end, maximum_requests=6):
    """Fallback public daily OHLCV download when Yahoo is rate limited."""
    frames = []
    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    d2 = pd.Timestamp(end).strftime("%Y%m%d")
    for ticker in list(tickers)[: int(maximum_requests)]:
        if ticker.startswith("^"):
            continue
        symbol = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
        try:
            response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            from io import StringIO

            frame = pd.read_csv(StringIO(response.text))
            if frame.empty or "Date" not in frame.columns:
                continue
            frame["Date"] = pd.to_datetime(frame["Date"])
            frame = frame.set_index("Date").sort_index()
            frame.columns = [str(column).lower() for column in frame.columns]
            if not {"open", "high", "low", "close"}.issubset(frame.columns):
                continue
            frame["adj_close"] = frame["close"]
            if "volume" not in frame:
                frame["volume"] = 0.0
            alias = _alias(ticker)
            frame = frame[["open", "high", "low", "close", "adj_close", "volume"]].apply(pd.to_numeric, errors="coerce")
            frame.columns = [f"{alias}_{column}" for column in frame.columns]
            frames.append(frame)
        except (requests.RequestException, ValueError, pd.errors.ParserError):
            continue
    if not frames:
        raise RuntimeError("Stooq returned no usable fallback market data.")
    return pd.concat(frames, axis=1).sort_index()


def _download_fred_series(series_id, alias, start, end):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    from io import StringIO

    frame = pd.read_csv(StringIO(response.text))
    date_column = frame.columns[0]
    value_column = frame.columns[-1]
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    result = frame.set_index(date_column)[[value_column]].rename(columns={value_column: alias})
    return result.loc[start:end]


def _download_dbnomics_fred_series(series_id, alias, start, end):
    """Download the same FRED series through DBnomics' public mirror."""
    url = f"https://api.db.nomics.world/v22/series/FRED/{series_id}?observations=1"
    response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    payload = response.json()
    docs = payload.get("series", {}).get("docs", [])
    if not docs:
        raise ValueError(f"DBnomics returned no observations for FRED/{series_id}.")
    doc = docs[0]
    periods = doc.get("period", [])
    values = doc.get("value", [])
    if not periods or not values:
        raise ValueError(f"DBnomics returned an empty FRED/{series_id} series.")
    frame = pd.DataFrame({alias: pd.to_numeric(values, errors="coerce")}, index=pd.to_datetime(periods))
    return frame.loc[start:end]


def _download_macro_with_fallback(series_id, alias, start, end):
    try:
        return _download_fred_series(series_id, alias, start, end), "fred"
    except Exception as fred_error:
        try:
            return _download_dbnomics_fred_series(series_id, alias, start, end), "dbnomics_fred_mirror"
        except Exception as mirror_error:
            raise RuntimeError(f"FRED failed: {fred_error}; DBnomics mirror failed: {mirror_error}") from mirror_error


def load_market_and_macro_data(
    start="2015-01-01",
    end=None,
    tickers=None,
    include_credit_spreads=True,
    refresh=False,
):
    """Return an aligned dataset and explicit provenance/warning metadata."""
    ensure_directories()
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = list(dict.fromkeys(tickers or DEFAULT_TICKERS))
    if "QQQ" not in tickers:
        tickers.insert(0, "QQQ")
    market_cache = RAW_DIR / "market_ohlcv.csv"
    download_status_path = RAW_DIR / "market_download_status.json"
    warnings = []

    market = None if refresh else _read_cached(market_cache)
    available_before = set()
    if market is not None:
        available_before = {ticker for ticker in tickers if f"{_alias(ticker)}_close" in market.columns}
    missing_before = [ticker for ticker in tickers if ticker not in available_before]
    previous_status = {}
    if download_status_path.exists():
        try:
            previous_status = json.loads(download_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_status = {}
    last_attempt = pd.to_datetime(previous_status.get("attempted_at"), errors="coerce")
    cooldown_active = (
        not refresh
        and pd.notna(last_attempt)
        and pd.Timestamp.now(tz=None) - last_attempt.tz_localize(None) < pd.Timedelta(hours=6)
    )
    needs_download = market is None or market.index.min() > pd.Timestamp(start) or bool(missing_before)
    if needs_download and not cooldown_active:
        requested_download = tickers if market is None else missing_before
        try:
            downloaded_market = _download_yfinance(requested_download, start, end)
            market = downloaded_market if market is None else market.combine_first(downloaded_market)
            market.to_csv(market_cache)
            status = {"attempted_at": pd.Timestamp.now().isoformat(), "requested": requested_download, "success": True}
        except Exception as error:
            market = _read_cached(market_cache)
            try:
                stooq_market = _download_stooq(requested_download, start, end)
                market = stooq_market if market is None else market.combine_first(stooq_market)
                warnings.append(f"Yahoo download failed; Stooq fallback was used: {error}")
            except Exception as stooq_error:
                warnings.append(f"Yahoo and Stooq downloads failed: {error}; {stooq_error}")
            if market is None:
                legacy = Path(__file__).resolve().parents[1] / "data" / "raw" / "combined_market_data.csv"
                market = _read_cached(legacy)
            if market is None:
                raise RuntimeError("Market download failed and no cache is available.") from error
            warnings.append(f"Market download failed; cached data used: {error}")
            market.to_csv(market_cache)
            status = {
                "attempted_at": pd.Timestamp.now().isoformat(), "requested": requested_download,
                "success": False, "error": str(error),
            }
        download_status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    elif needs_download and cooldown_active:
        warnings.append("Some requested tickers are missing; a failed/recent download is in the six-hour retry cooldown.")

    market = market.loc[start:end].copy()
    if "qqq_close" not in market:
        raise ValueError("Aligned market data does not contain qqq_close.")
    qqq_index = market["qqq_close"].dropna().index
    combined = market.reindex(qqq_index)
    downloaded_fred = []
    missing_fred = []

    if include_credit_spreads:
        macro_frames = []
        download_tasks = {}
        credit_sources = {}
        for series_id, alias in FRED_SERIES.items():
            cache_path = RAW_DIR / f"fred_{series_id}.csv"
            frame = None if refresh else _read_cached(cache_path)
            if frame is not None:
                macro_frames.append(frame.rename(columns={frame.columns[0]: alias}))
            else:
                download_tasks[series_id] = (alias, cache_path)

        if download_tasks:
            with ThreadPoolExecutor(max_workers=min(6, len(download_tasks))) as executor:
                futures = {
                    executor.submit(_download_macro_with_fallback, series_id, alias, start, end): (series_id, alias, cache_path)
                    for series_id, (alias, cache_path) in download_tasks.items()
                }
                for future in as_completed(futures):
                    series_id, alias, cache_path = futures[future]
                    try:
                        frame, source_name = future.result()
                        frame.to_csv(cache_path)
                        downloaded_fred.append(series_id)
                        credit_sources[series_id] = source_name
                        macro_frames.append(frame)
                    except Exception as error:
                        missing_fred.append(series_id)
                        warnings.append(f"FRED {series_id} unavailable: {error}")
        if macro_frames:
            macro = pd.concat(macro_frames, axis=1).sort_index().reindex(qqq_index).ffill(limit=10)
            combined = pd.concat([combined, macro], axis=1)

    combined = combined.sort_index().ffill(limit=5)
    combined.to_csv(RAW_DIR / "aligned_market_macro.csv")
    provenance = {
        "start": str(combined.index.min().date()),
        "end": str(combined.index.max().date()),
        "rows": int(len(combined)),
        "columns": int(combined.shape[1]),
        "tickers_requested": tickers,
        "tickers_available": [ticker for ticker in tickers if f"{_alias(ticker)}_close" in combined.columns],
        "tickers_missing": [ticker for ticker in tickers if f"{_alias(ticker)}_close" not in combined.columns],
        "credit_series_requested": list(FRED_SERIES) if include_credit_spreads else [],
        "credit_series_downloaded_now": downloaded_fred,
        "credit_series_missing": missing_fred,
        "credit_series_sources": credit_sources if include_credit_spreads else {},
        "credit_data_note": "ICE BofA/FRED option-adjusted spreads are public credit-spread indices, not single-name CDS quotes.",
        "warnings": warnings,
    }
    pd.Series(provenance, dtype=object).to_json(RAW_DIR / "data_provenance.json", indent=2)
    return combined, provenance

"""Paths and reproducible defaults for the QQQ blending pipeline."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

SEED = 42
TARGET_TICKER = "QQQ"
# Locked development target: a next-day QQQ return above 1.5%.
TARGET_THRESHOLD = 0.015
SUBSET_TRAIN_RATIO = 0.60
HOLDOUT_RATIO = 0.20
FINAL_VALIDATION_RATIO = 0.20
PURGE_SIZE = 1

DEFAULT_TICKERS = [
    "QQQ", "SPY", "IWM", "DIA", "TLT", "SHY", "HYG", "LQD",
    "GLD", "USO", "UUP", "EEM", "XLK", "XLF", "XLV", "XLE", "^VIX",
]

FRED_SERIES = {
    "BAMLH0A0HYM2": "credit_hy_oas",
    "BAMLC0A0CM": "credit_corporate_oas",
    "BAMLC0A4CBBB": "credit_bbb_oas",
    "NFCI": "financial_conditions_index",
    "T10Y2Y": "treasury_10y_2y_spread",
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "DFF": "fed_funds_rate",
}


def ensure_directories():
    for path in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)

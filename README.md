# QQQ Blending Ensemble Pipeline

This repository implements a chronological blending ensemble for short-horizon
QQQ event prediction. It contains source code and tests only: market data,
feature caches, trained models, and generated results are created locally and
are intentionally excluded from version control.

## Architecture and split

```text
All chronological labelled rows
|- 60% SubsetTrain: fit/tune LR, SVC, KNN
|- 20% Holdout: generate three base probabilities; fit/tune XGBoost meta-model
`- 20% Final Validation: one final classification evaluation and backtest
```

One observation is purged before each later block because the target uses the
next close. For exactly 1,000 rows the nominal boundaries are 600/200/200; the
effective fitted rows are 599/199/200 after the two one-row purges.

The target is:

```text
1  if next-day QQQ close-to-close return > 1.50%
0  otherwise, including returns equal to or below 1.50%
```

## Feature engineering

The initial universe must contain at least 500 predictors. The locked report run
creates 828 from lagged returns, momentum, volatility,
distribution statistics, volume, cross-asset correlation/beta, public credit
spread proxies, rates, financial conditions, technical indicators and calendar
features. `pandas-ta`, pandas rolling methods and sklearn algorithms are called
directly; existing indicators and optimization algorithms are not recoded.

The selection sequence is fitted on SubsetTrain only:

1. Hourglass filter: missingness/variance checks plus target correlation and
   mutual information.
2. Chronological univariate metric filter: ROC-AUC, accuracy, balanced accuracy,
   precision, recall or F1. A configurable strict lower threshold may be
   applied (for example, ROC-AUC > 0.55) before ranking the surviving features.
3. Multicollinearity pruning followed by ExtraTrees or L1-logistic embedded
   importance. sklearn RFE is applied when the remaining dimension exceeds the
   configured trigger.

Train-only EDA assigns each retained feature to RobustScaler, MinMaxScaler,
StandardScaler or imputation-only passthrough inside a ColumnTransformer.

FRED ICE BofA option-adjusted spread series are public credit-risk proxies, not
single-name CDS quotations. Missing series are downloaded and cached; failures
are explicitly recorded in `data/raw/data_provenance.json`.

### Wavelet denoising

Wavelet preprocessing is optional and is applied only to predictor inputs. Each
date is reconstructed from a trailing window ending on that date; raw QQQ
prices remain the source of the target, realised return and backtest P&L.

It can process OHLC prices, OHLC plus volume, or (as an explicitly experimental
choice) those market fields plus signed macro/credit proxies. Prices are handled
in log space, volume in log1p space, and signed macro series in native units.
The reference paper applies its cleaning to its daily stock dataset and says its
technical indicators come from open, high, low, close and volume; it does not
document a separate macro/credit data treatment.

Two threshold rules are available:

- `universal_mad` (default): robust MAD noise estimation on the finest detail
  level and the Donoho-Johnstone universal threshold.
- `paper_std`: paper-inspired, per-detail-level coefficient standard-deviation
  thresholds with conventional small-coefficient shrinkage.

The paper *A comprehensive evaluation of ensemble learning for stock-market
prediction* states on page 11 that wavelet coefficients are screened using
their standard deviation and then inverse-transformed, but does not specify the
mother wavelet, decomposition scales, threshold direction, shrinkage mode or a
causal protocol. It also says to remove coefficients above the standard
deviation, which would remove large jumps/signals rather than small noise. The
project records this ambiguity and does not reproduce that direction literally.
It uses an invertible DWT implementation because the paper's displayed CWT
formula is not accompanied by the scales or inverse-transform procedure needed
for an exact CWT reproduction.

## Run

Create a local environment and install the pinned dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run the pipeline from the repository root:

```powershell
.\.venv\Scripts\python.exe run_feature_engineering.py
.\.venv\Scripts\python.exe run_feature_engineering.py --wavelet --wavelet-threshold-rule paper_std
.\.venv\Scripts\python.exe run_feature_engineering.py --wavelet --n-jobs 4
.\.venv\Scripts\python.exe run_feature_engineering.py --metric roc_auc --stage2-min-metric 0.55
.\.venv\Scripts\python.exe run_model_training.py
.\.venv\Scripts\python.exe run_model_training.py --optimized-preset --optimize-threshold --cost-bps 10
.\.venv\Scripts\python.exe run_model_training.py --optimized-preset --optimize-threshold --cost-bps 10 --evaluation-mode walk_forward --walk-forward-frequency quarterly
.\run_app.ps1
```

The Streamlit application contains exactly two screens:

- Feature Engineering
- Model Training & Prediction

Within Feature Engineering, execute the workflow in order. Each step persists
its artefacts, so later steps reuse earlier work rather than rerunning it:

1. Data and optional causal wavelet denoising.
2. Cached 500+ feature-universe construction.
3. Stage 1: train-only EDA and hourglass filter.
4. Stage 2: chronological univariate metrics and any strict score threshold.
5. Stage 3: multicollinearity pruning, embedded selection and optional RFE.

The model screen remains unavailable until Stage 3 has produced the final
dataset and fitted feature transformer.

The model screen can either use a fixed trading threshold or select it from
chronological out-of-fold meta-model probabilities on Blend Holdout.  The
automatic threshold search never reads Final Validation labels and records
its candidate table in `threshold_optimization.csv`.

All generated files remain local under `data/` and `outputs/` and are excluded
from the Git repository.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_newversion
```

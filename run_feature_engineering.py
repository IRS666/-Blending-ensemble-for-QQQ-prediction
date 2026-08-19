"""CLI entry point for data acquisition, 500+ features, EDA, and selection."""

import argparse
import json

from data_pipeline import load_market_and_macro_data
from feature_pipeline import create_feature_universe
from feature_selection import SelectionConfig, run_feature_selection
from nvconfig import TABLES_DIR, TARGET_THRESHOLD
from split_utils import chronological_blending_split
from wavelet_pipeline import WaveletConfig, denoise_market_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--target-threshold", type=float, default=TARGET_THRESHOLD)
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--stage2-min-metric", type=float, default=0.0)
    parser.add_argument("--final-features", type=int, default=40)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--wavelet", action="store_true")
    parser.add_argument("--wavelet-name", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=2)
    parser.add_argument(
        "--wavelet-threshold-rule",
        choices=("universal_mad", "paper_std"),
        default="universal_mad",
    )
    parser.add_argument("--wavelet-window", type=int, default=128)
    parser.add_argument("--wavelet-minimum-history", type=int, default=32)
    parser.add_argument("--wavelet-threshold-scale", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1, help="-1 uses up to eight CPU workers.")
    args = parser.parse_args()

    data, provenance = load_market_and_macro_data(start=args.start, refresh=args.refresh)
    feature_source = data
    wavelet_summary = {"enabled": False}
    if args.wavelet:
        feature_source, wavelet_summary, _ = denoise_market_data(
            data,
            WaveletConfig(
                wavelet=args.wavelet_name,
                level=args.wavelet_level,
                threshold_rule=args.wavelet_threshold_rule,
                window_size=args.wavelet_window,
                minimum_history=args.wavelet_minimum_history,
                threshold_scale=args.wavelet_threshold_scale,
                n_jobs=args.n_jobs,
            ),
        )
    universe, cache_info = create_feature_universe(
        feature_source,
        target_threshold=args.target_threshold,
        target_data=data,
        use_cache=not args.no_feature_cache,
        return_cache_info=True,
    )
    split = chronological_blending_split(universe)
    config = SelectionConfig(
        stage2_metric=args.metric,
        stage2_min_metric=args.stage2_min_metric,
        final_feature_count=args.final_features,
    )
    config.n_jobs = args.n_jobs
    _, _, summary = run_feature_selection(universe, split["subset_train"].index, config)
    print(json.dumps({"data": provenance, "wavelet": wavelet_summary, "factor_cache": cache_info, "selection": summary}, indent=2, default=str))
    print(f"Outputs: {TABLES_DIR}")


if __name__ == "__main__":
    main()

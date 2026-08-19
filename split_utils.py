"""Exact nominal 60/20/20 chronological blending split with boundary purge."""

import math

from nvconfig import FINAL_VALIDATION_RATIO, HOLDOUT_RATIO, PURGE_SIZE, SUBSET_TRAIN_RATIO


def chronological_blending_split(
    frame,
    subset_ratio=SUBSET_TRAIN_RATIO,
    holdout_ratio=HOLDOUT_RATIO,
    validation_ratio=FINAL_VALIDATION_RATIO,
    purge_size=PURGE_SIZE,
):
    total = float(subset_ratio + holdout_ratio + validation_ratio)
    if abs(total - 1.0) > 1e-9:
        raise ValueError("SubsetTrain + Holdout + Final Validation ratios must sum to 1.")
    n = len(frame)
    subset_end = math.floor(n * subset_ratio)
    holdout_end = math.floor(n * (subset_ratio + holdout_ratio))
    purge_size = int(purge_size)
    if subset_end - purge_size <= 0 or holdout_end - purge_size <= subset_end:
        raise ValueError("Not enough rows for the requested purge and split ratios.")
    return {
        "subset_train": frame.iloc[: subset_end - purge_size].copy(),
        "holdout": frame.iloc[subset_end : holdout_end - purge_size].copy(),
        "final_validation": frame.iloc[holdout_end:].copy(),
        "purged_subset_boundary": frame.iloc[subset_end - purge_size : subset_end].copy(),
        "purged_holdout_boundary": frame.iloc[holdout_end - purge_size : holdout_end].copy(),
        "nominal_counts": {
            "subset_train": subset_end,
            "holdout": holdout_end - subset_end,
            "final_validation": n - holdout_end,
        },
    }


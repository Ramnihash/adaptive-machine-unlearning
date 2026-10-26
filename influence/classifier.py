"""Influence tier classification (low, medium, high)."""

import numpy as np

DEFAULT_THRESHOLDS = {"low": 0.33, "high": 0.66}


def _validate_thresholds(thresholds: dict[str, float]) -> tuple[float, float]:
    """Validate thresholds dictionary and return (low, high)."""
    if "low" not in thresholds or "high" not in thresholds:
        raise ValueError("thresholds must contain 'low' and 'high' keys.")

    low = float(thresholds["low"])
    high = float(thresholds["high"])

    if not (0.0 <= low <= high <= 1.0):
        raise ValueError(
            f"Invalid thresholds: expected 0.0 <= low <= high <= 1.0, got low={low}, high={high}."
        )

    return low, high


def classify_tier(
    score: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Classify an influence score into 'low', 'medium', or 'high'.

    Classification rules:
        score <= low       -> "low"
        low < score <= high -> "medium"
        score > high       -> "high"

    Args:
        score: Influence score in [0.0, 1.0].
        thresholds: Dictionary with "low" and "high" keys. Defaults to low=0.33, high=0.66.

    Returns:
        One of "low", "medium", or "high".
    """
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"score must be between 0.0 and 1.0, got {score}.")

    low, high = _validate_thresholds(thresholds if thresholds is not None else DEFAULT_THRESHOLDS)

    if score <= low:
        return "low"
    if score <= high:
        return "medium"
    return "high"


def classify_user_tiers(
    scores: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Classify an array of influence scores into tiers.

    Args:
        scores: 1D array of influence scores in [0.0, 1.0].
        thresholds: Dictionary with "low" and "high" keys. Defaults to low=0.33, high=0.66.

    Returns:
        List of string tiers corresponding to each score in scores.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")

    low, high = _validate_thresholds(thresholds if thresholds is not None else DEFAULT_THRESHOLDS)

    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("All scores must be between 0.0 and 1.0.")

    tiers = []
    for s in scores:
        if s <= low:
            tiers.append("low")
        elif s <= high:
            tiers.append("medium")
        else:
            tiers.append("high")
    return tiers

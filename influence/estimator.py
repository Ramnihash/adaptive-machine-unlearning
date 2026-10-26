"""User influence estimation based on training interactions (CP2.1)."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from data.split import DataSplit


@dataclass(frozen=True)
class UserInfluence:
    """Calculated influence metrics and tier for a target user."""

    user_idx: int
    degree: int
    score: float
    tier: str  # "low" | "medium" | "high"


def compute_user_degrees(
    train: pd.DataFrame,
    num_users: int,
) -> np.ndarray:
    """Extract interaction count for each user in 0..num_users-1 using training interactions only.

    Args:
        train: DataFrame containing at least a "user_idx" column.
        num_users: Total number of users.

    Returns:
        1D np.ndarray of shape (num_users,) with non-negative integer degrees.
    """
    if num_users <= 0:
        raise ValueError("num_users must be positive.")

    degrees = np.zeros(num_users, dtype=np.int64)
    if train.empty:
        return degrees

    if "user_idx" not in train.columns:
        raise ValueError("train DataFrame must contain a 'user_idx' column.")

    user_indices = train["user_idx"].to_numpy(dtype=np.int64)
    if (user_indices < 0).any() or (user_indices >= num_users).any():
        raise ValueError(f"user_idx values must be within 0..{num_users - 1}.")

    counts = np.bincount(user_indices, minlength=num_users)
    return counts[:num_users]


def compute_influence_scores(
    degrees: np.ndarray,
    method: str = "percentile",
) -> np.ndarray:
    """Map user interaction degrees to deterministic scores in [0.0, 1.0].

    Equal training-interaction degrees produce equal scores, independent
    of row ordering or user index.

    Edge cases:
    - If len(degrees) == 1, or all users have equal degrees (min == max),
      returns 0.5 for all users.
    - If len(degrees) == 0, raises ValueError.

    Args:
        degrees: 1D array of non-negative integer degrees for all users.
        method: "percentile" (default, mid-rank ECDF) or "minmax" (linear degree scaling).

    Returns:
        1D np.ndarray of shape (len(degrees),) with values in [0.0, 1.0].
    """
    degrees = np.asarray(degrees)
    if degrees.ndim != 1:
        raise ValueError("degrees must be a 1D array.")

    num_users = len(degrees)
    if num_users == 0:
        raise ValueError("degrees array cannot be empty.")

    if (degrees < 0).any():
        raise ValueError("degrees must be non-negative.")

    min_deg = int(degrees.min())
    max_deg = int(degrees.max())

    # When there is only one user or all users have equal degrees, assign neutral 0.5.
    if num_users == 1 or min_deg == max_deg:
        return np.full(num_users, 0.5, dtype=np.float64)

    if method == "percentile":
        # Mid-rank ECDF: score = (c_less + 0.5 * c_equal) / N
        # Strictly deterministic, equal degrees get equal scores.
        vals, counts = np.unique(degrees, return_counts=True)
        cum_less = np.zeros(len(vals), dtype=np.float64)
        cum_less[1:] = np.cumsum(counts)[:-1]
        val_to_score = (cum_less + 0.5 * counts) / float(num_users)
        idx = np.searchsorted(vals, degrees)
        return val_to_score[idx]

    if method == "minmax":
        # Linear min-max scaling into [0.0, 1.0]
        return (degrees - min_deg) / float(max_deg - min_deg)

    raise ValueError(f"Unknown scoring method {method!r}. Choose 'percentile' or 'minmax'.")

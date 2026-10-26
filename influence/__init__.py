"""User influence estimation and tier classification package."""

from data.split import DataSplit
from influence.classifier import (
    DEFAULT_THRESHOLDS,
    classify_tier,
    classify_user_tiers,
)
from influence.estimator import (
    UserInfluence,
    compute_influence_scores,
    compute_user_degrees,
)


def estimate_influence(
    split: DataSplit,
    target_user: int | None = None,
    thresholds: dict[str, float] | None = None,
    method: str = "percentile",
) -> UserInfluence | dict[int, UserInfluence]:
    """Calculate user influence scores and tiers using training interactions only.

    Validation and test interactions are never accessed, preserving data isolation.

    Args:
        split: The DataSplit containing split.train and split.num_users.
        target_user: Optional user_idx (0-based) to return influence for a single user.
                     If None, returns a dictionary mapping all user_idx -> UserInfluence.
        thresholds: Tier thresholds with "low" and "high" keys. Defaults to {"low": 0.33, "high": 0.66}.
        method: Scoring method ("percentile" or "minmax").

    Returns:
        UserInfluence if target_user is given, else dict[int, UserInfluence].
    """
    if split.num_users <= 0:
        raise ValueError("num_users must be positive.")

    if target_user is not None:
        if not (0 <= target_user < split.num_users):
            raise ValueError(
                f"target_user {target_user} is outside 0..{split.num_users - 1}."
            )

    degrees = compute_user_degrees(split.train, split.num_users)
    scores = compute_influence_scores(degrees, method=method)
    tiers = classify_user_tiers(scores, thresholds=thresholds)

    if target_user is not None:
        return UserInfluence(
            user_idx=target_user,
            degree=int(degrees[target_user]),
            score=float(scores[target_user]),
            tier=tiers[target_user],
        )

    return {
        u: UserInfluence(
            user_idx=u,
            degree=int(degrees[u]),
            score=float(scores[u]),
            tier=tiers[u],
        )
        for u in range(split.num_users)
    }


__all__ = [
    "DEFAULT_THRESHOLDS",
    "UserInfluence",
    "classify_tier",
    "classify_user_tiers",
    "compute_influence_scores",
    "compute_user_degrees",
    "estimate_influence",
]

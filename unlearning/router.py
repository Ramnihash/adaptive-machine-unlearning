"""Adaptive unlearning router (CP13/CP14)."""

from dataclasses import dataclass
from typing import cast

from data.split import DataSplit
from influence import UserInfluence, estimate_influence

VALID_TIERS = {"low", "medium", "high"}

DEFAULT_STRATEGY_MAP: dict[str, str] = {
    "low": "gradient_ascent",
    "medium": "localized_update",
    "high": "targeted_retraining",
}


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable routing decision produced by the adaptive unlearning router."""

    target_user: int
    influence: UserInfluence
    strategy: str


def select_strategy(
    tier: str,
    strategy_map: dict[str, str] | None = None,
) -> str:
    """Map an influence tier ('low', 'medium', 'high') to a strategy name.

    Args:
        tier: One of 'low', 'medium', or 'high'.
        strategy_map: Mapping from tier to strategy name. Defaults to DEFAULT_STRATEGY_MAP.

    Returns:
        The selected strategy name string.

    Raises:
        ValueError: If tier is invalid, strategy_map is empty/invalid, or
                    strategy_map lacks a non-empty strategy for the tier.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"Unknown tier {tier!r}. Expected one of {sorted(VALID_TIERS)}.")

    mapping = DEFAULT_STRATEGY_MAP if strategy_map is None else strategy_map

    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("strategy_map must be a non-empty dictionary.")

    if tier not in mapping:
        raise ValueError(f"Strategy mapping has no entry for tier {tier!r}.")

    strategy = mapping[tier]
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError(f"Strategy mapping for tier {tier!r} must be a non-empty string.")

    return strategy.strip()


def route_unlearning(
    split: DataSplit,
    target_user: int,
    strategy_map: dict[str, str] | None = None,
    thresholds: dict[str, float] | None = None,
    scoring_method: str = "percentile",
) -> RoutingDecision:
    """Calculate target user influence and select an unlearning strategy.

    Does not modify data, mutate the model, train anything, or execute
    the strategy.

    Args:
        split: DataSplit containing training interactions and num_users.
        target_user: Target user index (0-based integer).
        strategy_map: Optional mapping of tier -> strategy name.
        thresholds: Optional dict with 'low' and 'high' thresholds passed
                    through to the influence estimator.
        scoring_method: Scoring method ('percentile' or 'minmax').

    Returns:
        RoutingDecision with target_user, computed UserInfluence, and selected strategy.
    """
    influence = cast(
        UserInfluence,
        estimate_influence(
            split,
            target_user=target_user,
            thresholds=thresholds,
            method=scoring_method,
        ),
    )

    strategy = select_strategy(influence.tier, strategy_map=strategy_map)

    return RoutingDecision(
        target_user=target_user,
        influence=influence,
        strategy=strategy,
    )

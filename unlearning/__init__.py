"""Adaptive machine unlearning module."""

from unlearning.pipeline import (
    adaptive_unlearning_stage,
    run_adaptive_unlearning,
    unlearn,
)
from unlearning.router import (
    DEFAULT_STRATEGY_MAP,
    RoutingDecision,
    route_unlearning,
    select_strategy,
)
from unlearning.strategies.gradient_ascent import gradient_ascent_unlearning
from unlearning.strategies.localized_update import (
    get_affected_neighborhood,
    localized_update_unlearning,
)
from unlearning.strategies.targeted_retraining import targeted_retraining_unlearning
from unlearning.strategies.warm_start import warm_start_retraining

__all__ = [
    "DEFAULT_STRATEGY_MAP",
    "RoutingDecision",
    "adaptive_unlearning_stage",
    "get_affected_neighborhood",
    "gradient_ascent_unlearning",
    "localized_update_unlearning",
    "route_unlearning",
    "run_adaptive_unlearning",
    "select_strategy",
    "targeted_retraining_unlearning",
    "unlearn",
    "warm_start_retraining",
]

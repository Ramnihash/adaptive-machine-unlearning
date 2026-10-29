"""Unlearning strategy implementations."""

from unlearning.strategies.gradient_ascent import gradient_ascent_unlearning
from unlearning.strategies.localized_update import (
    get_affected_neighborhood,
    localized_update_unlearning,
)
from unlearning.strategies.targeted_retraining import targeted_retraining_unlearning
from unlearning.strategies.warm_start import warm_start_retraining

__all__ = [
    "get_affected_neighborhood",
    "gradient_ascent_unlearning",
    "localized_update_unlearning",
    "targeted_retraining_unlearning",
    "warm_start_retraining",
]

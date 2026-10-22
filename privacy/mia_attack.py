"""Attack classifier (CP5.3)."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

ATTACK_CONFIG = {
    "classifier": "GradientBoostingClassifier",
    "n_estimators": 100,
    "max_depth": 3,
    "random_state": 42,
}


def train_attack(features: np.ndarray, labels: np.ndarray) -> GradientBoostingClassifier:
    classifier = GradientBoostingClassifier(**{k: v for k, v in ATTACK_CONFIG.items() if k != "classifier"})
    return classifier.fit(features, labels)


def membership_probability(classifier: GradientBoostingClassifier, features: np.ndarray) -> np.ndarray:
    return classifier.predict_proba(features)[:, 1]


def attack_metrics(classifier: GradientBoostingClassifier, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probability = membership_probability(classifier, features)
    return {
        "auc": float(roc_auc_score(labels, probability)),
        "accuracy": float(accuracy_score(labels, probability >= 0.5)),
        "n": int(len(labels)),
    }

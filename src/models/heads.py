"""Lightweight prediction heads trained on top of pre-computed embeddings.

The DINOv3 + head pipeline (and later the stacking meta-learner) trains
small models on the parquet feature cache instead of fine-tuning the
backbone.  This keeps every (asset_type, attribute) head independent
and cheap to retrain.

Available heads:
- ``logistic``   sklearn LogisticRegression, class-balanced.
- ``mlp``        sklearn MLPClassifier / Regressor.
- ``knn``        sklearn KNeighborsClassifier / Regressor.
- ``ridge``      sklearn Ridge regressor / RidgeClassifier.
- ``catboost``   CatBoost classifier / regressor (optional;
                 falls back to logistic / ridge if catboost is missing).
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


def _has_catboost() -> bool:
    try:
        import catboost  # noqa: F401

        return True
    except ImportError:
        return False


def make_classifier(head: str, *, random_state: int = 42) -> Any:
    """Return a fitted classification estimator for the named head."""
    head = head.lower()
    if head == "logistic":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
    if head == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=200,
            random_state=random_state,
        )
    if head == "knn":
        return KNeighborsClassifier(n_neighbors=11, weights="distance", metric="cosine")
    if head == "ridge":
        return RidgeClassifier(class_weight="balanced", random_state=random_state)
    if head == "catboost":
        if not _has_catboost():
            return make_classifier("logistic", random_state=random_state)
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_state=random_state,
            verbose=False,
        )
    raise ValueError(f"Unknown classification head: {head}")


def make_regressor(head: str, *, random_state: int = 42) -> Any:
    """Return a regression estimator for the named head."""
    head = head.lower()
    if head == "ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    if head == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=200,
            random_state=random_state,
        )
    if head == "knn":
        return KNeighborsRegressor(n_neighbors=11, weights="distance", metric="cosine")
    if head == "catboost":
        if not _has_catboost():
            return make_regressor("ridge", random_state=random_state)
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_state=random_state,
            verbose=False,
        )
    if head == "logistic":
        # Allow "logistic" as a synonym for ridge for callers that don't track
        # the classification/regression distinction.
        return make_regressor("ridge", random_state=random_state)
    raise ValueError(f"Unknown regression head: {head}")


__all__ = ["make_classifier", "make_regressor"]

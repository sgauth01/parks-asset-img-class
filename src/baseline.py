"""Simple baseline models for the BC Parks image attribute project.

Two dummy predictors that establish a "no-image-info" floor for every
attribute in the schema:

- :class:`MajorityClassPredictor` for categorical / boolean / ordinal-bin
  attributes.  Equivalent to
  ``sklearn.dummy.DummyClassifier(strategy="most_frequent")`` but keeps
  the fitted majority class easy to inspect and log.
- :class:`MedianRegressor` for numeric and count attributes.  Equivalent
  to ``sklearn.dummy.DummyRegressor(strategy="median")``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MajorityClassPredictor:
    """Predict the most common training label for every row."""

    def __init__(self) -> None:
        self.fitted_value_: Any | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: Any, y: Any) -> "MajorityClassPredictor":
        """Fit the majority label from ``y``.

        ``X`` is accepted for sklearn-style compatibility and is not used.
        """
        labels = pd.Series(y).dropna()
        if labels.empty:
            raise ValueError("MajorityClassPredictor requires at least one label.")

        counts = labels.value_counts()
        self.fitted_value_ = counts.index[0]
        self.classes_ = np.array(sorted(labels.unique(), key=str))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the fitted majority label for every row in ``X``."""
        if self.fitted_value_ is None:
            raise RuntimeError("MajorityClassPredictor must be fit before predict.")
        return np.repeat(self.fitted_value_, len(X))


class MedianRegressor:
    """Predict the training median for every row.

    Equivalent to ``sklearn.dummy.DummyRegressor(strategy="median")`` but
    kept here so the fitted constant is easy to MLflow-log alongside the
    other baseline runs.
    """

    def __init__(self) -> None:
        self.fitted_value_: float | None = None

    def fit(self, X: Any, y: Any) -> "MedianRegressor":
        del X
        s = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
        if s.empty:
            raise ValueError("MedianRegressor requires at least one numeric value.")
        self.fitted_value_ = float(s.median())
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.fitted_value_ is None:
            raise RuntimeError("MedianRegressor must be fit before predict.")
        return np.repeat(self.fitted_value_, len(X))


__all__ = ["MajorityClassPredictor", "MedianRegressor"]

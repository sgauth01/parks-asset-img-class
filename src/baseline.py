"""Simple baseline models for the BC Parks image attribute project."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MajorityClassPredictor:
    """Predict the most common training label for every row.

    This is equivalent to scikit-learn's ``DummyClassifier(strategy="most_frequent")``,
    but keeps the fitted majority class easy to inspect and log.
    """

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


__all__ = ["MajorityClassPredictor"]

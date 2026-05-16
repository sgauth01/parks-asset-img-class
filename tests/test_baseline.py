"""Tests for simple distribution-only baseline models."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline import cross_validate_majority_class, evaluate_majority_class


def test_majority_class_baseline_reports_imbalance_and_scores() -> None:
    train = pd.DataFrame({"target": ["wood", "wood", "steel", None]})
    test = pd.DataFrame({"target": ["wood", "steel", None]})

    results = evaluate_majority_class(train, test, targets=["target"])

    assert len(results) == 1
    row = results.iloc[0]
    assert row["prediction"] == "wood"
    assert row["n_train_labels"] == 3
    assert row["n_test_labels"] == 2
    assert row["train_majority_share"] == 2 / 3
    assert row["accuracy"] == 0.5


def test_cross_validated_majority_class_uses_train_data_only() -> None:
    train = pd.DataFrame(
        {
            "target": [
                "wood",
                "wood",
                "wood",
                "steel",
                "wood",
                "steel",
                None,
            ]
        }
    )

    summary, folds = cross_validate_majority_class(
        train, targets=["target"], n_splits=3, random_state=7
    )

    assert len(summary) == 1
    assert len(folds) == 3
    row = summary.iloc[0]
    assert row["attribute"] == "target"
    assert row["strategy"] == "majority_class_cv"
    assert row["n_folds"] == 3
    assert row["n_labels"] == 6
    assert 0 <= row["accuracy_mean"] <= 1
    assert 0 <= row["macro_f1_mean"] <= 1

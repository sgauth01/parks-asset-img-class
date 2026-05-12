"""Tests for issue #11 majority-class classification baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baseline_classification import (  # noqa: E402
    clean_target_frame,
    run_target_baseline,
    split_by_asset,
)
from src.baseline import MajorityClassPredictor  # noqa: E402


def test_majority_class_predictor_repeats_most_frequent_label() -> None:
    model = MajorityClassPredictor().fit(np.zeros((5, 1)), ["wood", "steel", "wood"])

    assert model.fitted_value_ == "wood"
    assert model.predict(np.zeros((3, 1))).tolist() == ["wood", "wood", "wood"]


def test_majority_class_predictor_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        MajorityClassPredictor().predict(np.zeros((1, 1)))


def test_clean_target_frame_removes_missing_placeholders() -> None:
    df = pd.DataFrame(
        {
            "asset_id": [1, 2, 3, 4],
            "target": ["Timber", "TBD", "", None],
        }
    )

    cleaned = clean_target_frame(df, "target")

    assert cleaned["target"].tolist() == ["Timber"]


def test_split_by_asset_keeps_assets_in_only_one_split() -> None:
    df = pd.DataFrame(
        {
            "asset_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "target": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    )

    train, test = split_by_asset(df, test_size=0.5, split_seed=42)

    assert set(train["asset_id"]).isdisjoint(set(test["asset_id"]))
    assert len(train) + len(test) == len(df)


def test_run_target_baseline_returns_expected_metrics() -> None:
    df = pd.DataFrame(
        {
            "asset_id": list(range(10)),
            "target": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )

    result = run_target_baseline(
        df,
        "target",
        test_size=0.3,
        split_seed=7,
        data_version="test",
        log_to_mlflow=False,
    )

    assert result["target"] == "target"
    assert result["majority_class"] in {"a", "b"}
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert 0.0 <= result["weighted_f1"] <= 1.0

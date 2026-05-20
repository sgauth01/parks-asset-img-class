"""Tests for issue #11 majority-class / median baselines on the new split."""

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
)
from src.baseline import MajorityClassPredictor, MedianRegressor  # noqa: E402


def test_majority_class_predictor_repeats_most_frequent_label() -> None:
    model = MajorityClassPredictor().fit(np.zeros((5, 1)), ["wood", "steel", "wood"])
    assert model.fitted_value_ == "wood"
    assert model.predict(np.zeros((3, 1))).tolist() == ["wood", "wood", "wood"]


def test_majority_class_predictor_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        MajorityClassPredictor().predict(np.zeros((1, 1)))


def test_median_regressor_returns_training_median() -> None:
    model = MedianRegressor().fit(None, [1.0, 2.0, 3.0, 100.0])
    assert model.fitted_value_ == pytest.approx(2.5)
    assert model.predict(np.zeros((4, 1))).tolist() == [2.5, 2.5, 2.5, 2.5]


def test_median_regressor_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        MedianRegressor().predict(np.zeros((1, 1)))


def test_clean_target_frame_removes_missing_placeholders() -> None:
    df = pd.DataFrame(
        {
            "asset_id": [1, 2, 3, 4],
            "target": ["Timber", "TBD", "", None],
        }
    )
    cleaned = clean_target_frame(df, "target")
    assert cleaned["target"].tolist() == ["Timber"]


def _toy_attribute_file(
    train_dir: Path,
    attribute_column: str,
    rows: list[dict],
) -> None:
    df = pd.DataFrame(rows)
    # File name convention: comma in column name becomes underscore.
    safe = attribute_column.replace(",_", "_").replace(",", "_")
    df.to_csv(train_dir / f"{safe}_train.csv", index=False)


def test_run_target_baseline_classification_on_synthetic_split(tmp_path: Path) -> None:
    """Classification path on a synthetic per-attribute file."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    rows = []
    for a in range(40):
        for img in range(2):
            rows.append(
                {
                    "image_path": f"img_{a}_{img}.jpg",
                    "asset_id": a,
                    "profile_name": "Trail Bridge",
                    "attr_decking_material": "Timber" if a % 4 != 0 else "Steel",
                }
            )
    _toy_attribute_file(train_dir, "attr_decking_material", rows)

    result = run_target_baseline(
        "attr_decking_material",
        test_size=0.2,
        split_seed=48,
        data_version="test",
        train_dir=train_dir,
        log_to_mlflow=False,
    )
    assert result["attribute"] == "attr_decking_material"
    assert result["predictor"] == "majority_class"
    assert result["fitted_value"] == "Timber"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert 0.0 <= result["weighted_f1"] <= 1.0
    assert result["n_train_assets"] + result["n_val_assets"] == 40


def test_run_target_baseline_numeric_on_synthetic_split(tmp_path: Path) -> None:
    """Numeric / median path on a synthetic per-attribute file."""
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    rows = []
    for a in range(40):
        rows.append(
            {
                "image_path": f"img_{a}.jpg",
                "asset_id": a,
                "profile_name": "Trail Bridge",
                "attr_length": float(a),  # 0..39, median 19.5
            }
        )
    _toy_attribute_file(train_dir, "attr_length", rows)

    result = run_target_baseline(
        "attr_length",
        test_size=0.2,
        split_seed=48,
        data_version="test",
        train_dir=train_dir,
        log_to_mlflow=False,
    )
    assert result["attribute"] == "attr_length"
    assert result["predictor"] == "median"
    assert isinstance(result["fitted_value"], float)
    assert result["rmse"] >= 0.0
    assert result["mae"] >= 0.0


def test_run_target_baseline_rejects_unknown_attribute(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    with pytest.raises(ValueError, match="Unknown attribute column"):
        run_target_baseline(
            "attr_does_not_exist",
            test_size=0.15,
            split_seed=48,
            data_version="test",
            train_dir=train_dir,
            log_to_mlflow=False,
        )

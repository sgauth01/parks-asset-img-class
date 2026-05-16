"""Tests for the per-attribute (new suggested split) 85/15 splitter."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.per_attribute_splits import (  # noqa: E402
    ATTRIBUTE_COLUMNS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_SIZE,
    _filename_for,
    iter_attribute_splits,
    load_per_attribute_file,
    load_per_attribute_train_val,
    split_train_val,
)


def _toy(asset_ids: list[int]) -> pd.DataFrame:
    rows = []
    for a in asset_ids:
        for img in range(2):  # 2 images per asset, like the real data
            rows.append(
                {
                    "image_path": f"img_{a}_{img}.jpg",
                    "asset_id": a,
                    "profile_name": "Trail Bridge",
                    "attr_decking_material": "Timber" if a % 2 == 0 else "Steel",
                }
            )
    return pd.DataFrame(rows)


def test_filename_for_handles_comma_attribute() -> None:
    assert _filename_for("attr_decking_material") == "attr_decking_material_train.csv"
    assert _filename_for("attr_material_frame,_tank,_body") == "attr_material_frame_tank_body_train.csv"


def test_split_is_asset_grouped() -> None:
    df = _toy(list(range(20)))
    train, val = split_train_val(df, test_size=0.2, random_state=42)
    train_assets = set(train["asset_id"])
    val_assets = set(val["asset_id"])
    assert not (train_assets & val_assets), "asset leakage between train/val"
    assert len(train_assets) + len(val_assets) == 20


def test_split_is_deterministic() -> None:
    df = _toy(list(range(40)))
    a1, b1 = split_train_val(df, test_size=0.15, random_state=DEFAULT_SPLIT_SEED)
    a2, b2 = split_train_val(df, test_size=0.15, random_state=DEFAULT_SPLIT_SEED)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(b1, b2)


def test_load_per_attribute_file_round_trip(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    df = _toy(list(range(10)))
    df.to_csv(train_dir / "attr_decking_material_train.csv", index=False)
    loaded = load_per_attribute_file("attr_decking_material", train_dir=train_dir)
    assert len(loaded) == len(df)


def test_load_per_attribute_train_val_returns_85_15(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    df = _toy(list(range(100)))
    df.to_csv(train_dir / "attr_decking_material_train.csv", index=False)
    train, val = load_per_attribute_train_val(
        "attr_decking_material",
        train_dir=train_dir,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_SPLIT_SEED,
    )
    val_asset_frac = val["asset_id"].nunique() / 100
    assert 0.10 <= val_asset_frac <= 0.20, val_asset_frac


def test_iter_attribute_splits_skips_missing(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    df = _toy(list(range(20)))
    df.to_csv(train_dir / "attr_decking_material_train.csv", index=False)
    out = list(iter_attribute_splits(train_dir=train_dir))
    # Only one CSV present so we should get exactly one entry.
    assert [r[0] for r in out] == ["attr_decking_material"]
    attr, train, val = out[0]
    assert not (set(train["asset_id"]) & set(val["asset_id"]))


def test_attribute_columns_match_schema() -> None:
    from src.data.schema import load_schema

    schema = load_schema()
    expected = set(schema.attribute_columns())
    assert set(ATTRIBUTE_COLUMNS) == expected

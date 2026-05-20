"""Tests for the DINOv3 + head pipeline on the per-attribute 85/15 split.

Uses a deterministic fake feature cache so heads have an obvious signal
to learn — no real backbone download.  The real backbone is smoke-tested
locally via ``scripts/build_features.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embed.dinov3 import (  # noqa: E402
    FeatureCache,
    extract_features_for_split,
    load_features,
    save_features,
)
from src.models.heads import make_classifier, make_regressor  # noqa: E402


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_dinov3_new_split",
        REPO_ROOT / "scripts" / "run_dinov3_new_split.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


runner = _load_runner_module()


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def test_make_classifier_handles_known_heads() -> None:
    for head in ["logistic", "mlp", "knn", "ridge"]:
        assert make_classifier(head) is not None
    assert make_classifier("catboost") is not None  # falls back if catboost missing


def test_make_regressor_handles_known_heads() -> None:
    for head in ["ridge", "mlp", "knn", "logistic"]:
        assert make_regressor(head) is not None
    assert make_regressor("catboost") is not None


def test_make_classifier_rejects_unknown_head() -> None:
    with pytest.raises(ValueError):
        make_classifier("not_a_head")


# ---------------------------------------------------------------------------
# FeatureCache
# ---------------------------------------------------------------------------


def test_feature_cache_dim_and_features() -> None:
    df = pd.DataFrame(
        {"image_path": ["a.jpg", "b.jpg"], "f_0": [1.0, 0.0], "f_1": [0.0, 1.0]}
    )
    cache = FeatureCache(df=df, model_id="fake")
    assert cache.dim == 2
    assert cache.features().shape == (2, 2)


def test_feature_cache_aligned_to_returns_nan_for_missing() -> None:
    df = pd.DataFrame(
        {"image_path": ["a.jpg", "b.jpg"], "f_0": [1.0, 0.0], "f_1": [0.0, 1.0]}
    )
    cache = FeatureCache(df=df, model_id="fake")
    feats, missing = cache.aligned_to(["a.jpg", "missing.jpg", "b.jpg"])
    assert feats.shape == (3, 2)
    assert missing.tolist() == [False, True, False]
    assert (feats[0] == np.array([1.0, 0.0])).all()
    assert (feats[2] == np.array([0.0, 1.0])).all()
    assert np.isnan(feats[1]).all()


def test_feature_cache_aligned_to_dedupes_input_cache() -> None:
    df = pd.DataFrame(
        {
            "image_path": ["a.jpg", "a.jpg", "b.jpg"],
            "f_0": [1.0, 9.0, 0.0],
            "f_1": [0.0, 9.0, 1.0],
        }
    )
    cache = FeatureCache(df=df, model_id="fake")
    feats, missing = cache.aligned_to(["a.jpg", "b.jpg"])
    assert missing.tolist() == [False, False]
    assert (feats[0] == np.array([1.0, 0.0])).all()  # first occurrence kept


# ---------------------------------------------------------------------------
# Feature extraction / round-trip
# ---------------------------------------------------------------------------


def _fake_features(images):
    out = np.zeros((len(images), 4), dtype=np.float32)
    for i in range(len(images)):
        out[i] = [1.0, 0.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 1.0, 0.0, 0.0]
    return out


def test_extract_features_with_fake_extractor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    monkeypatch.setattr(
        "src.embed.dinov3._load_image",
        lambda path, repo_root=None: Image.new("RGB", (4, 4)),
    )
    df = pd.DataFrame({"image_path": [f"img_{i}.jpg" for i in range(4)]})
    cache = extract_features_for_split(df, extractor=_fake_features, batch_size=2)
    assert len(cache.df) == 4 and cache.dim == 4

    save_features(cache, out_dir=tmp_path)
    reloaded = load_features(cache.model_id, out_dir=tmp_path)
    assert reloaded.dim == 4 and len(reloaded.df) == 4


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def test_resolve_column_handles_no_commas_filename() -> None:
    df = pd.DataFrame({"asset_id": [1], "attr_decking_material": ["Timber"]})
    assert runner.resolve_column(df, "attr_decking_material") == "attr_decking_material"


def test_resolve_column_handles_comma_column() -> None:
    df = pd.DataFrame(
        {"asset_id": [1], "attr_material_frame,_tank,_body": ["Steel"]}
    )
    assert (
        runner.resolve_column(df, "attr_material_frame_tank_body")
        == "attr_material_frame,_tank,_body"
    )


def test_asset_grouped_split_no_leakage() -> None:
    df = pd.DataFrame(
        {
            "image_path": [f"img_{i}.jpg" for i in range(40)],
            "asset_id": list(range(20)) * 2,  # 2 images per asset
        }
    )
    train, val = runner.asset_grouped_split(df, test_size=0.2, random_state=42)
    assert not (set(train["asset_id"]) & set(val["asset_id"]))
    assert len(train["asset_id"].unique()) + len(val["asset_id"].unique()) == 20


# ---------------------------------------------------------------------------
# run_attribute end-to-end with synthetic features
# ---------------------------------------------------------------------------


def _make_cls_train_csv(tmp_path: Path) -> Path:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    rows = []
    for asset_id in range(20):
        label = "Timber" if asset_id % 2 == 0 else "Steel"
        for img_idx in range(2):
            rows.append(
                {
                    "image_path": f"asset_{asset_id}_img_{img_idx}.jpg",
                    "asset_id": asset_id,
                    "attr_decking_material": label,
                }
            )
    pd.DataFrame(rows).to_csv(train_dir / "attr_decking_material_train.csv", index=False)
    return train_dir


def _signal_cache_for(train_csv: Path) -> FeatureCache:
    df = pd.read_csv(train_csv)
    feat_rows = []
    for _, row in df.iterrows():
        is_timber = row["attr_decking_material"] == "Timber"
        vec = [1.0, 0.0, 0.0, 0.0] if is_timber else [0.0, 1.0, 0.0, 0.0]
        feat_rows.append({"image_path": row["image_path"], **{f"f_{i}": v for i, v in enumerate(vec)}})
    return FeatureCache(df=pd.DataFrame(feat_rows), model_id="fake/cls")


def test_run_attribute_cls_learns_perfect_signal(tmp_path: Path) -> None:
    train_dir = _make_cls_train_csv(tmp_path)
    cache = _signal_cache_for(train_dir / "attr_decking_material_train.csv")

    result = runner.run_attribute(
        file_key="attr_decking_material",
        kind="cls",
        cache=cache,
        train_dir=train_dir,
        head="logistic",
        numeric_head="ridge",
        test_size=0.2,
        split_seed=42,
        max_assets=None,
    )
    assert result is not None
    assert result["metrics"]["macro_f1"] == pytest.approx(1.0)
    assert result["metrics"]["accuracy"] == pytest.approx(1.0)
    assert result["n_val_assets"] >= 2


def test_run_attribute_num_returns_finite_rmse(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for asset_id in range(30):
        truth = 1.0 if asset_id % 2 == 0 else 5.0
        for img_idx in range(2):
            rows.append(
                {
                    "image_path": f"asset_{asset_id}_img_{img_idx}.jpg",
                    "asset_id": asset_id,
                    "attr_length": truth + rng.normal(0, 0.01),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(train_dir / "attr_length_train.csv", index=False)

    feat_rows = []
    for _, row in df.iterrows():
        vec = [1.0, 0.0, 0.0, 0.0] if row["attr_length"] < 3 else [0.0, 1.0, 0.0, 0.0]
        feat_rows.append({"image_path": row["image_path"], **{f"f_{i}": v for i, v in enumerate(vec)}})
    cache = FeatureCache(df=pd.DataFrame(feat_rows), model_id="fake/num")

    result = runner.run_attribute(
        file_key="attr_length",
        kind="num",
        cache=cache,
        train_dir=train_dir,
        head="logistic",
        numeric_head="ridge",
        test_size=0.2,
        split_seed=42,
        max_assets=None,
    )
    assert result is not None
    assert np.isfinite(result["metrics"]["rmse"])
    assert np.isfinite(result["metrics"]["mae"])
    # Ridge shrinks toward mean but ranking should still be correct
    assert result["metrics"]["rmse"] < 3.0


def test_run_attribute_returns_none_for_missing_file(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    cache = FeatureCache(df=pd.DataFrame({"image_path": [], "f_0": []}), model_id="fake")
    result = runner.run_attribute(
        file_key="attr_decking_material",
        kind="cls",
        cache=cache,
        train_dir=train_dir,
        head="logistic",
        numeric_head="ridge",
        test_size=0.2,
        split_seed=42,
        max_assets=None,
    )
    assert result is None

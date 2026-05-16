"""Per-attribute 85/15 splits matching the new suggested `data/processed/train/*` files.

Background
----------
The shared train/test split (``data/processed/train.csv`` / ``test.csv``)
that drives :func:`src.data.splits.load_split` is image-level and applies
to every attribute uniformly.  The new suggested split uses a different
paradigm:

- One ``attr_X_train.csv`` per attribute (12 attribute columns + 4
  bin columns = 16 files), each containing only the rows where the
  attribute has a usable label.
- The held-out 15% test rows are **not** in the repo; we generate a
  validation set in-process via :class:`sklearn.model_selection.GroupShuffleSplit`
  on ``asset_id``.
- Random seed = 48; group key = ``asset_id``.

Public API
----------

>>> from src.data.per_attribute_splits import load_per_attribute_train_val
>>> train_df, val_df = load_per_attribute_train_val("attr_decking_material")

For pipelines that train one classifier per attribute, the natural
loop is:

>>> for attr, train_df, val_df in iter_attribute_splits():
...     ...
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_TRAIN_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "train"
DEFAULT_TEST_SIZE = 0.15
DEFAULT_SPLIT_SEED = 48

# Filename convention used by the new split: 12 attribute files + 4 bin
# files.  The 4 bin files are intentionally omitted from training by
# default (per the partner note: "we're only using 12 for now").
ATTRIBUTE_COLUMNS = [
    "attr_abutment_material",
    "attr_bridge_type",
    "attr_decking_material",
    "attr_fall_height",
    "attr_has_edge_guard",
    "attr_has_pedestrian_railing",
    "attr_length",
    "attr_material_frame,_tank,_body",
    "attr_number_of_steps",
    "attr_structure_material",
    "attr_structure_position",
    "attr_width",
]

BIN_COLUMNS = [
    "length_bin",
    "width_bin",
    "fall_height_bin",
    "steps_bin",
]


def _filename_for(attribute_column: str) -> str:
    """Map a CSV column name to the corresponding per-attribute train filename."""
    safe = attribute_column.replace(",_", "_").replace(",", "_")
    return f"{safe}_train.csv"


def load_per_attribute_file(
    attribute_column: str,
    train_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Read the new suggested per-attribute train CSV (no split applied yet)."""
    base = Path(train_dir) if train_dir is not None else DEFAULT_TRAIN_DIR
    p = base / _filename_for(attribute_column)
    if not p.exists():
        raise FileNotFoundError(
            f"Per-attribute train file not found: {p}.  "
            "Make sure your branch contains the data-split files."
        )
    return pd.read_csv(p)


def split_train_val(
    df: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SPLIT_SEED,
    group_col: str = "asset_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asset-grouped 85/15 split via :class:`GroupShuffleSplit`."""
    if df.empty:
        return df.copy(), df.copy()
    if df[group_col].nunique() < 2:
        # Degenerate — return everything as train, empty as val.
        return df.copy(), df.iloc[0:0].copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df[group_col].values))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def load_per_attribute_train_val(
    attribute_column: str,
    *,
    train_dir: str | Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_85, val_15) for one attribute, split by asset_id."""
    df = load_per_attribute_file(attribute_column, train_dir=train_dir)
    return split_train_val(df, test_size=test_size, random_state=random_state)


def iter_attribute_splits(
    attribute_columns: list[str] | None = None,
    *,
    train_dir: str | Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SPLIT_SEED,
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(attribute_column, train_85, val_15)`` for every attribute."""
    cols = attribute_columns or ATTRIBUTE_COLUMNS
    for col in cols:
        try:
            train_df, val_df = load_per_attribute_train_val(
                col,
                train_dir=train_dir,
                test_size=test_size,
                random_state=random_state,
            )
        except FileNotFoundError as exc:
            print(f"[skip] {col}: {exc}")
            continue
        yield col, train_df, val_df


__all__ = [
    "ATTRIBUTE_COLUMNS",
    "BIN_COLUMNS",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_TEST_SIZE",
    "iter_attribute_splits",
    "load_per_attribute_file",
    "load_per_attribute_train_val",
    "split_train_val",
]

"""Run DINOv3 / DINOv2 + classifier head on the per-attribute 85/15 split.

For each ``data/processed/train/attr_X_train.csv``:
1. Asset-grouped 85/15 holdout (``GroupShuffleSplit``, seed=48).
2. Look up cached DINOv3 features (parquet from ``scripts/build_features.py``).
3. Fit head: logistic for classification, ridge for numeric.
4. Predict on val rows, write predictions + log MLflow run.

Usage:
    python scripts/build_features.py --model facebook/dinov3-vitl16-pretrain-lvd1689m
    python scripts/run_dinov3_new_split.py
    python scripts/run_dinov3_new_split.py --no-mlflow
    python scripts/run_dinov3_new_split.py --max-assets 8       # smoke test
    python scripts/run_dinov3_new_split.py --model facebook/dinov2-large
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from src.embed import DEFAULT_DINOV3_MODEL, load_features, slug_for_model  # noqa: E402
from src.models.heads import make_classifier, make_regressor  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_SEED = 48
DEFAULT_TEST_SIZE = 0.15
MISSING_LABELS = {"", "nan", "none", "null", "tbd", "unknown"}

# Per issue #41: 12 attributes only (the 4 *_bin columns are intentionally
# excluded from this run).  File name keys (no commas).
ATTRIBUTE_KINDS: dict[str, str] = {
    "attr_abutment_material": "cls",
    "attr_bridge_type": "cls",
    "attr_decking_material": "cls",
    "attr_fall_height": "num",
    "attr_has_edge_guard": "cls",
    "attr_has_pedestrian_railing": "cls",
    "attr_length": "num",
    "attr_material_frame_tank_body": "cls",
    "attr_number_of_steps": "num",
    "attr_structure_material": "cls",
    "attr_structure_position": "cls",
    "attr_width": "num",
}


def _normalize(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def resolve_column(frame: pd.DataFrame, file_key: str) -> str:
    """Match a file-name key (no commas) to the actual column name (may have commas)."""
    if file_key in frame.columns:
        return file_key
    target = _normalize(file_key)
    matches = [c for c in frame.columns if _normalize(c) == target]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not resolve column for {file_key!r} in columns {list(frame.columns)[:20]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_DINOV3_MODEL, help="HuggingFace model id (must match the feature cache).")
    p.add_argument("--head", default="logistic", choices=["logistic", "ridge", "mlp", "knn", "catboost"], help="Classification head.")
    p.add_argument("--numeric-head", default="ridge", choices=["ridge", "mlp", "knn", "catboost"], help="Numeric head.")
    p.add_argument("--train-dir", type=Path, default=Path("data/processed/train"))
    p.add_argument("--features-dir", type=Path, default=Path("data/features"))
    p.add_argument("--predictions-dir", type=Path, default=Path("data/predictions/new_split"))
    p.add_argument("--metrics-csv", type=Path, default=Path("data/predictions/new_split/per_attribute_metrics.csv"))
    p.add_argument("--feature-suffix", default="", help="Optional suffix matching --suffix in build_features.py.")
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--max-assets", type=int, default=None, help="Smoke-test cap on val asset count per attribute.")
    p.add_argument("--attributes", nargs="*", default=list(ATTRIBUTE_KINDS), help="Subset of attribute file keys to run.")
    p.add_argument("--data-version", default="per-attribute-85-15")
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args()


def asset_grouped_split(df: pd.DataFrame, *, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asset-grouped 85/15 split. Same contract as PR #38's loader (seed=48)."""
    if df["asset_id"].nunique() < 2:
        return df.copy(), df.iloc[0:0].copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df["asset_id"].values))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _clean_class_labels(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    mask = out.notna() & ~out.str.lower().isin(MISSING_LABELS)
    return out.loc[mask]


def fit_and_predict_cls(
    *,
    train_feats: np.ndarray,
    val_feats: np.ndarray,
    y_train: pd.Series,
    head: str,
    random_state: int,
) -> np.ndarray:
    """Fit a classification head and return string predictions for the val features."""
    model = make_classifier(head, random_state=random_state)
    y_arr = np.asarray(y_train.tolist(), dtype=object)
    model.fit(train_feats, y_arr)
    preds = np.asarray(model.predict(val_feats), dtype=object)
    if preds.ndim > 1:
        preds = preds.reshape(-1)
    return preds


def fit_and_predict_num(
    *,
    train_feats: np.ndarray,
    val_feats: np.ndarray,
    y_train: np.ndarray,
    head: str,
    random_state: int,
) -> np.ndarray:
    """Fit a regression head and return numeric predictions for the val features."""
    model = make_regressor(head, random_state=random_state)
    model.fit(train_feats, y_train)
    return np.asarray(model.predict(val_feats), dtype=float)


def classification_metrics(y_true: Iterable[Any], y_pred: Iterable[Any]) -> dict[str, float]:
    y_true_a = np.asarray(list(y_true), dtype=object)
    y_pred_a = np.asarray(list(y_pred), dtype=object)
    return {
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "macro_f1": float(f1_score(y_true_a, y_pred_a, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_a, y_pred_a, average="weighted", zero_division=0)),
    }


def numeric_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_a = np.asarray(list(y_true), dtype=float)
    y_pred_a = np.asarray(list(y_pred), dtype=float)
    mse = mean_squared_error(y_true_a, y_pred_a)
    return {
        "mae": float(mean_absolute_error(y_true_a, y_pred_a)),
        "rmse": float(math.sqrt(mse)),
    }


def run_attribute(
    *,
    file_key: str,
    kind: str,
    cache,
    train_dir: Path,
    head: str,
    numeric_head: str,
    test_size: float,
    split_seed: int,
    max_assets: int | None,
) -> dict[str, Any] | None:
    csv_path = train_dir / f"{file_key}_train.csv"
    if not csv_path.exists():
        logger.warning("skip %s: %s missing", file_key, csv_path)
        return None
    df = pd.read_csv(csv_path)
    col = resolve_column(df, file_key)

    train_df, val_df = asset_grouped_split(df, test_size=test_size, random_state=split_seed)
    if max_assets is not None and not val_df.empty:
        keep_ids = val_df["asset_id"].drop_duplicates().head(max_assets).tolist()
        val_df = val_df[val_df["asset_id"].isin(keep_ids)].reset_index(drop=True)

    train_feats, train_missing = cache.aligned_to(train_df["image_path"].tolist())
    val_feats, val_missing = cache.aligned_to(val_df["image_path"].tolist())
    train_keep = ~train_missing
    val_keep = ~val_missing
    train_df = train_df.loc[train_keep].reset_index(drop=True)
    val_df = val_df.loc[val_keep].reset_index(drop=True)
    train_feats = train_feats[train_keep]
    val_feats = val_feats[val_keep]

    if len(train_df) == 0 or len(val_df) == 0:
        logger.warning("skip %s: empty after feature alignment", file_key)
        return None

    if kind == "cls":
        y_train_clean = _clean_class_labels(train_df[col])
        if y_train_clean.empty or y_train_clean.nunique() < 2:
            logger.warning("skip %s: <2 distinct training classes", file_key)
            return None
        train_mask = train_df.index.isin(y_train_clean.index)
        x_train = train_feats[train_mask]
        y_val_clean = _clean_class_labels(val_df[col])
        if y_val_clean.empty:
            logger.warning("skip %s: no val labels", file_key)
            return None
        val_mask = val_df.index.isin(y_val_clean.index)
        x_val = val_feats[val_mask]
        val_df_kept = val_df.loc[val_mask].reset_index(drop=True)
        y_pred = fit_and_predict_cls(
            train_feats=x_train,
            val_feats=x_val,
            y_train=y_train_clean,
            head=head,
            random_state=42,
        )
        metrics = classification_metrics(y_val_clean.tolist(), y_pred)
        row_records = pd.DataFrame({
            "image_path": val_df_kept["image_path"].values,
            "asset_id": val_df_kept["asset_id"].values,
            "attribute": file_key,
            "y_true": y_val_clean.values,
            "y_pred": y_pred,
        })
    else:  # numeric
        y_train_num = pd.to_numeric(train_df[col], errors="coerce")
        train_mask = y_train_num.notna()
        if train_mask.sum() < 5:
            logger.warning("skip %s: <5 numeric training rows", file_key)
            return None
        y_val_num = pd.to_numeric(val_df[col], errors="coerce")
        val_mask = y_val_num.notna()
        if val_mask.sum() == 0:
            logger.warning("skip %s: no numeric val rows", file_key)
            return None
        y_pred = fit_and_predict_num(
            train_feats=train_feats[train_mask.values],
            val_feats=val_feats[val_mask.values],
            y_train=y_train_num[train_mask].values,
            head=numeric_head,
            random_state=42,
        )
        metrics = numeric_metrics(y_val_num[val_mask].values, y_pred)
        val_df_kept = val_df.loc[val_mask].reset_index(drop=True)
        row_records = pd.DataFrame({
            "image_path": val_df_kept["image_path"].values,
            "asset_id": val_df_kept["asset_id"].values,
            "attribute": file_key,
            "y_true": y_val_num[val_mask].values,
            "y_pred": y_pred,
        })

    return {
        "attribute": file_key,
        "column": col,
        "kind": kind,
        "head": head if kind == "cls" else numeric_head,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df_kept)),
        "n_train_assets": int(train_df["asset_id"].nunique()),
        "n_val_assets": int(val_df_kept["asset_id"].nunique()),
        "metrics": metrics,
        "predictions": row_records,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    mlflow = None
    if not args.no_mlflow:
        try:
            import mlflow as _mlflow
            from src.mlflow_utils import make_run_name, make_standard_tags, setup_mlflow
            setup_mlflow()
            mlflow = _mlflow
        except ModuleNotFoundError:
            logger.warning("mlflow not installed; continuing without MLflow logging")
            mlflow = None

    cache = load_features(model_id=args.model, out_dir=args.features_dir, suffix=args.feature_suffix)
    logger.info("Loaded feature cache: %d rows, %d-d (%s)", len(cache.df), cache.dim, cache.model_id)

    model_slug = slug_for_model(args.model)
    pred_filename = f"{model_slug}__{args.head}_{args.numeric_head}__new15.csv"
    pred_path = args.predictions_dir / pred_filename
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    for file_key in args.attributes:
        if file_key not in ATTRIBUTE_KINDS:
            logger.warning("skip unknown attribute %s", file_key)
            continue
        result = run_attribute(
            file_key=file_key,
            kind=ATTRIBUTE_KINDS[file_key],
            cache=cache,
            train_dir=args.train_dir,
            head=args.head,
            numeric_head=args.numeric_head,
            test_size=args.test_size,
            split_seed=args.split_seed,
            max_assets=args.max_assets,
        )
        if result is None:
            continue
        all_predictions.append(result["predictions"])
        metric_row = {
            "pipeline": "dinov3_head",
            "model_name": model_slug,
            "head": result["head"],
            "attribute": file_key,
            "kind": result["kind"],
            "n_train_rows": result["n_train_rows"],
            "n_val_rows": result["n_val_rows"],
            "n_train_assets": result["n_train_assets"],
            "n_val_assets": result["n_val_assets"],
            **result["metrics"],
        }
        metrics_rows.append(metric_row)

        if result["kind"] == "cls":
            print(
                f"{file_key}: macro_f1={result['metrics']['macro_f1']:.3f} "
                f"accuracy={result['metrics']['accuracy']:.3f} "
                f"(n_val={result['n_val_rows']}, head={result['head']})"
            )
        else:
            print(
                f"{file_key}: RMSE={result['metrics']['rmse']:.3f} "
                f"MAE={result['metrics']['mae']:.3f} "
                f"(n_val={result['n_val_rows']}, head={result['head']})"
            )

        if mlflow is not None:
            task = f"T3_{file_key}"
            tags = make_standard_tags(
                task=task,
                model_family="dinov3",
                model_name=f"{model_slug}__{result['head']}",
                data_version=args.data_version,
                split_seed=args.split_seed,
                extra={"attribute_kind": result["kind"], "split": "per_attribute_85_15"},
            )
            with mlflow.start_run(run_name=make_run_name(task, f"{model_slug}_{result['head']}"), tags=tags):
                mlflow.log_param("attribute", file_key)
                mlflow.log_param("attribute_column", result["column"])
                mlflow.log_param("head", result["head"])
                mlflow.log_param("model_id", args.model)
                mlflow.log_param("n_train_rows", result["n_train_rows"])
                mlflow.log_param("n_val_rows", result["n_val_rows"])
                mlflow.log_param("n_train_assets", result["n_train_assets"])
                mlflow.log_param("n_val_assets", result["n_val_assets"])
                mlflow.log_param("test_size", args.test_size)
                mlflow.log_param("split_random_state", args.split_seed)
                for k, v in result["metrics"].items():
                    mlflow.log_metric(k, v)

    if not all_predictions:
        raise SystemExit("No attributes scored.")

    pd.concat(all_predictions, ignore_index=True).to_csv(pred_path, index=False)
    print(f"\nWrote predictions to {pred_path}")

    metrics_df = pd.DataFrame(metrics_rows)
    if args.metrics_csv.exists():
        existing = pd.read_csv(args.metrics_csv)
        keep_mask = ~((existing["pipeline"] == "dinov3_head") & (existing["model_name"] == model_slug))
        existing = existing.loc[keep_mask]
        merged = pd.concat([existing, metrics_df], ignore_index=True)
    else:
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        merged = metrics_df
    merged.to_csv(args.metrics_csv, index=False)
    print(f"Updated metrics in {args.metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

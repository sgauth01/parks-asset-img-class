"""Run dummy baselines on the new per-attribute splits.

Originally written for issue #11 (majority-class classification baseline);
this version is upgraded to:

1. Use the locked attribute list from ``configs/schema.yaml`` instead of
   a hand-maintained ``DEFAULT_TARGETS`` constant.
2. Read the new partner-supplied per-attribute splits in
   ``data/processed/train/attr_X_train.csv`` and generate validation
   subsplits via ``GroupShuffleSplit(test_size=0.15, random_state=48)``
   keyed by ``asset_id`` (the new split contract).
3. Predict the **majority class** for categorical / boolean / ordinal-bin
   attributes and the **median** for numeric / count attributes.  Each
   attribute gets its own (train_85, val_15) sub-split.
4. Log per-attribute runs to MLflow (one run per attribute) and write a
   tidy summary CSV.

Usage:
    python scripts/run_baseline_classification.py
    python scripts/run_baseline_classification.py --no-mlflow
    python scripts/run_baseline_classification.py --attributes attr_decking_material attr_length
    python scripts/run_baseline_classification.py --cv 5    # asset-grouped K-fold CV
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline import MajorityClassPredictor, MedianRegressor  # noqa: E402
from src.data.per_attribute_splits import (  # noqa: E402
    ATTRIBUTE_COLUMNS,
    DEFAULT_CV_FOLDS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_SIZE,
    load_per_attribute_kfold,
    load_per_attribute_train_val,
)
from src.data.schema import AttributeKind, load_schema  # noqa: E402
from src.mlflow_utils import (  # noqa: E402
    classification_metrics,
    log_classification_run,
    log_numeric_run,
    make_run_name,
    make_standard_tags,
    numeric_metrics,
    setup_mlflow,
)

MISSING_LABELS = {"", "nan", "none", "null", "tbd", "unknown"}

CLS_KINDS = {
    AttributeKind.CATEGORICAL,
    AttributeKind.BOOLEAN,
    AttributeKind.ORDINAL_BIN,
}


def clean_target_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return rows with usable labels for one target.

    Used for classification targets only; numeric targets are coerced
    with ``pd.to_numeric`` instead.
    """
    required = ["asset_id", target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {target}: {missing}")

    out = df[required].copy()
    out[target] = out[target].astype("string").str.strip()
    mask = out[target].notna() & ~out[target].str.lower().isin(MISSING_LABELS)
    return out.loc[mask].reset_index(drop=True)


def run_target_baseline(
    attribute_column: str,
    *,
    test_size: float,
    split_seed: int,
    data_version: str,
    train_dir: Path | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, object]:
    """Fit and evaluate the dummy baseline for one attribute on the new split.

    The training rows come from the partner's per-attribute file
    (``data/processed/train/attr_X_train.csv``); the validation rows
    come from an in-process ``GroupShuffleSplit`` on ``asset_id``.
    """
    schema = load_schema()
    if attribute_column not in schema.attributes:
        raise ValueError(f"Unknown attribute column: {attribute_column}")
    kind = schema.kind_of(attribute_column)

    train_df, val_df = load_per_attribute_train_val(
        attribute_column,
        train_dir=train_dir,
        test_size=test_size,
        random_state=split_seed,
    )
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"{attribute_column} produced an empty train or val split "
            f"({len(train_df)} / {len(val_df)})."
        )

    if kind in CLS_KINDS:
        y_train_series = clean_target_frame(train_df, attribute_column)
        if y_train_series.empty:
            raise ValueError(f"{attribute_column}: no usable training labels.")
        y_train = y_train_series[attribute_column]
        model = MajorityClassPredictor().fit(None, y_train)

        y_val_series = clean_target_frame(val_df, attribute_column)
        if y_val_series.empty:
            raise ValueError(f"{attribute_column}: no usable validation labels.")
        y_pred = model.predict(y_val_series)
        metrics = classification_metrics(y_val_series[attribute_column], y_pred)
        result: dict[str, object] = {
            "attribute": attribute_column,
            "kind": kind.value,
            "predictor": "majority_class",
            "fitted_value": str(model.fitted_value_),
            "n_classes": int(y_train.nunique()),
            "n_train_rows": int(len(y_train_series)),
            "n_val_rows": int(len(y_val_series)),
            "n_train_assets": int(train_df["asset_id"].nunique()),
            "n_val_assets": int(val_df["asset_id"].nunique()),
            **metrics,
        }
        if log_to_mlflow:
            task = f"T3_{attribute_column}"
            run_id = log_classification_run(
                run_name=make_run_name(task, "majority_class"),
                tags=make_standard_tags(
                    task=task,
                    model_family="baseline",
                    model_name="majority_class",
                    data_version=data_version,
                    split_seed=split_seed,
                    extra={"attribute_kind": kind.value, "split": "per_attribute_85_15"},
                ),
                params={
                    "attribute": attribute_column,
                    "fitted_value": result["fitted_value"],
                    "n_classes": result["n_classes"],
                    "n_train_rows": result["n_train_rows"],
                    "n_val_rows": result["n_val_rows"],
                    "test_size": test_size,
                    "split_random_state": split_seed,
                },
                y_true=y_val_series[attribute_column],
                y_pred=y_pred,
            )
            result["mlflow_run_id"] = run_id
        return result

    # numeric / count
    y_train_num = pd.to_numeric(train_df[attribute_column], errors="coerce").dropna()
    if y_train_num.empty:
        raise ValueError(f"{attribute_column}: no usable numeric training values.")
    model = MedianRegressor().fit(None, y_train_num)

    val_keep = val_df.copy()
    val_keep[attribute_column] = pd.to_numeric(
        val_keep[attribute_column], errors="coerce"
    )
    val_keep = val_keep[val_keep[attribute_column].notna()].reset_index(drop=True)
    if val_keep.empty:
        raise ValueError(f"{attribute_column}: no usable numeric validation values.")

    y_pred = model.predict(val_keep)
    metrics = numeric_metrics(val_keep[attribute_column], y_pred)
    result = {
        "attribute": attribute_column,
        "kind": kind.value,
        "predictor": "median",
        "fitted_value": float(model.fitted_value_),  # type: ignore[arg-type]
        "n_classes": "",
        "n_train_rows": int(len(y_train_num)),
        "n_val_rows": int(len(val_keep)),
        "n_train_assets": int(train_df["asset_id"].nunique()),
        "n_val_assets": int(val_df["asset_id"].nunique()),
        **metrics,
    }
    if log_to_mlflow:
        task = f"T3_{attribute_column}"
        run_id = log_numeric_run(
            run_name=make_run_name(task, "median_regressor"),
            tags=make_standard_tags(
                task=task,
                model_family="baseline",
                model_name="median_regressor",
                data_version=data_version,
                split_seed=split_seed,
                extra={"attribute_kind": kind.value, "split": "per_attribute_85_15"},
            ),
            params={
                "attribute": attribute_column,
                "fitted_value": result["fitted_value"],
                "n_train_rows": result["n_train_rows"],
                "n_val_rows": result["n_val_rows"],
                "test_size": test_size,
                "split_random_state": split_seed,
            },
            y_true=val_keep[attribute_column],
            y_pred=y_pred,
        )
        result["mlflow_run_id"] = run_id
    return result


def _score_fold(
    attribute_column: str,
    kind: AttributeKind,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict[str, object] | None:
    """Fit baseline on train_df, evaluate on val_df.  Returns metrics dict or None if unusable."""
    if train_df.empty or val_df.empty:
        return None

    if kind in CLS_KINDS:
        y_train_series = clean_target_frame(train_df, attribute_column)
        y_val_series = clean_target_frame(val_df, attribute_column)
        if y_train_series.empty or y_val_series.empty:
            return None
        model = MajorityClassPredictor().fit(None, y_train_series[attribute_column])
        y_pred = model.predict(y_val_series)
        metrics = classification_metrics(y_val_series[attribute_column], y_pred)
        return {
            "predictor": "majority_class",
            "fitted_value": str(model.fitted_value_),
            "n_train_rows": int(len(y_train_series)),
            "n_val_rows": int(len(y_val_series)),
            "n_train_assets": int(train_df["asset_id"].nunique()),
            "n_val_assets": int(val_df["asset_id"].nunique()),
            **metrics,
        }

    y_train_num = pd.to_numeric(train_df[attribute_column], errors="coerce").dropna()
    if y_train_num.empty:
        return None
    model = MedianRegressor().fit(None, y_train_num)
    val_keep = val_df.copy()
    val_keep[attribute_column] = pd.to_numeric(val_keep[attribute_column], errors="coerce")
    val_keep = val_keep[val_keep[attribute_column].notna()].reset_index(drop=True)
    if val_keep.empty:
        return None
    y_pred = model.predict(val_keep)
    metrics = numeric_metrics(val_keep[attribute_column], y_pred)
    return {
        "predictor": "median",
        "fitted_value": float(model.fitted_value_),  # type: ignore[arg-type]
        "n_train_rows": int(len(y_train_num)),
        "n_val_rows": int(len(val_keep)),
        "n_train_assets": int(train_df["asset_id"].nunique()),
        "n_val_assets": int(val_df["asset_id"].nunique()),
        **metrics,
    }


def run_target_baseline_cv(
    attribute_column: str,
    *,
    n_splits: int,
    train_dir: Path | None = None,
) -> dict[str, object]:
    """Run asset-grouped K-fold CV for one attribute.  Returns per-fold rows + mean/std summary."""
    schema = load_schema()
    if attribute_column not in schema.attributes:
        raise ValueError(f"Unknown attribute column: {attribute_column}")
    kind = schema.kind_of(attribute_column)

    fold_rows: list[dict[str, object]] = []
    for fold_idx, train_df, val_df in load_per_attribute_kfold(
        attribute_column, train_dir=train_dir, n_splits=n_splits
    ):
        scored = _score_fold(attribute_column, kind, train_df, val_df)
        if scored is None:
            continue
        scored = {
            "attribute": attribute_column,
            "kind": kind.value,
            "fold": fold_idx,
            **scored,
        }
        fold_rows.append(scored)

    if not fold_rows:
        raise ValueError(f"{attribute_column}: no usable CV folds.")

    metric_keys = (
        ["accuracy", "macro_f1", "weighted_f1"]
        if kind in CLS_KINDS
        else ["mae", "rmse", "r2"]
    )
    aggregated: dict[str, object] = {
        "attribute": attribute_column,
        "kind": kind.value,
        "n_folds": len(fold_rows),
    }
    for k in metric_keys:
        values = [float(r[k]) for r in fold_rows if r.get(k) is not None]
        if values:
            aggregated[f"{k}_mean"] = statistics.fmean(values)
            aggregated[f"{k}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        else:
            aggregated[f"{k}_mean"] = None
            aggregated[f"{k}_std"] = None
    return {"folds": fold_rows, "summary": aggregated}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run majority-class / median baselines on the new per-attribute "
            "85/15 splits (issue #11, refreshed for the new split contract)."
        )
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/processed/train"),
        help="Directory containing the per-attribute train CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/classification_baselines.csv"),
        help="Where to write the baseline summary CSV.",
    )
    parser.add_argument(
        "--attributes",
        nargs="*",
        default=ATTRIBUTE_COLUMNS,
        help="Attribute columns to evaluate.  Defaults to all 12 schema attributes.",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--data-version", default="per-attribute-85-15")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help=(
            f"If set to N>=2, run asset-grouped K-fold CV with N folds instead "
            f"of the single 85/15 holdout. Suggested: {DEFAULT_CV_FOLDS}."
        ),
    )
    parser.add_argument(
        "--cv-folds-output",
        type=Path,
        default=Path("results/baselines/classification_baselines_cv_folds.csv"),
        help="Per-fold CV rows (only written when --cv is set).",
    )
    return parser.parse_args()


def _run_holdout(args: argparse.Namespace) -> int:
    if not args.no_mlflow:
        setup_mlflow()

    rows: list[dict[str, object]] = []
    for attr in args.attributes:
        try:
            result = run_target_baseline(
                attr,
                test_size=args.test_size,
                split_seed=args.split_seed,
                data_version=args.data_version,
                train_dir=args.train_dir,
                log_to_mlflow=not args.no_mlflow,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {attr}: {exc}")
            continue

        rows.append(result)
        if result["predictor"] == "majority_class":
            print(
                f"{attr}: majority={result['fitted_value']!r} "
                f"accuracy={result['accuracy']:.3f} macro_f1={result['macro_f1']:.3f} "
                f"(train_assets={result['n_train_assets']}, val_assets={result['n_val_assets']})"
            )
        else:
            print(
                f"{attr}: median={result['fitted_value']:.2f} "
                f"RMSE={result['rmse']:.3f} MAE={result['mae']:.3f} "
                f"(train_assets={result['n_train_assets']}, val_assets={result['n_val_assets']})"
            )

    if not rows:
        raise SystemExit("No baseline runs completed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nWrote {len(rows)} baseline rows to {args.output}")
    return 0


def _run_cv(args: argparse.Namespace) -> int:
    if args.cv < 2:
        raise SystemExit(f"--cv must be >= 2 (got {args.cv}).")

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for attr in args.attributes:
        try:
            out = run_target_baseline_cv(attr, n_splits=args.cv, train_dir=args.train_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {attr}: {exc}")
            continue
        fold_rows.extend(out["folds"])  # type: ignore[arg-type]
        s = out["summary"]
        summary_rows.append(s)  # type: ignore[arg-type]
        if "macro_f1_mean" in s and s["macro_f1_mean"] is not None:
            print(
                f"{attr} [cv={s['n_folds']}]: "  # type: ignore[index]
                f"macro_f1={s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f}, "  # type: ignore[index]
                f"acc={s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}"  # type: ignore[index]
            )
        else:
            print(
                f"{attr} [cv={s['n_folds']}]: "  # type: ignore[index]
                f"RMSE={s['rmse_mean']:.3f} ± {s['rmse_std']:.3f}, "  # type: ignore[index]
                f"MAE={s['mae_mean']:.3f} ± {s['mae_std']:.3f}"  # type: ignore[index]
            )

    if not summary_rows:
        raise SystemExit("No CV runs completed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.output, index=False)
    args.cv_folds_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.cv_folds_output, index=False)
    print(
        f"\nWrote CV summary ({len(summary_rows)} attrs) to {args.output}; "
        f"per-fold rows ({len(fold_rows)}) to {args.cv_folds_output}"
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.cv:
        return _run_cv(args)
    return _run_holdout(args)


if __name__ == "__main__":
    raise SystemExit(main())

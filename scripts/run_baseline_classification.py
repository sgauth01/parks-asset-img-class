"""Run majority-class classification baselines on the processed dataset.

This script addresses issue #11 by fitting a simple benchmark for each
categorical target and logging every run to MLflow.

Usage:
    python scripts/run_baseline_classification.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline import MajorityClassPredictor  # noqa: E402
from src.mlflow_utils import (  # noqa: E402
    classification_metrics,
    log_classification_run,
    make_run_name,
    make_standard_tags,
    setup_mlflow,
)


DEFAULT_TARGETS = [
    "profile_name",
    "attr_decking_material",
    "attr_has_edge_guard",
    "attr_has_pedestrian_railing",
    "attr_material_frame,_tank,_body",
    "attr_structure_material",
    "attr_structure_position",
    "length_bin",
    "width_bin",
    "fall_height_bin",
    "steps_bin",
]

MISSING_LABELS = {"", "nan", "none", "null", "tbd", "unknown"}


def clean_target_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return rows with usable labels for one target."""
    required = ["asset_id", target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {target}: {missing}")

    out = df[required].copy()
    out[target] = out[target].astype("string").str.strip()
    mask = out[target].notna() & ~out[target].str.lower().isin(MISSING_LABELS)
    return out.loc[mask].reset_index(drop=True)


def split_by_asset(
    df: pd.DataFrame,
    *,
    test_size: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by asset id to avoid leakage across images of the same asset."""
    asset_ids = pd.Series(df["asset_id"].dropna().unique())
    if len(asset_ids) < 2:
        raise ValueError("Need at least two assets to create a train/test split.")

    train_assets, test_assets = train_test_split(
        asset_ids,
        test_size=test_size,
        random_state=split_seed,
    )
    train = df[df["asset_id"].isin(train_assets)].copy()
    test = df[df["asset_id"].isin(test_assets)].copy()
    return train, test


def run_target_baseline(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float,
    split_seed: int,
    data_version: str,
    log_to_mlflow: bool = True,
) -> dict[str, object]:
    """Fit and evaluate one majority-class baseline."""
    target_df = clean_target_frame(df, target)
    train, test = split_by_asset(target_df, test_size=test_size, split_seed=split_seed)

    if train.empty or test.empty:
        raise ValueError(f"{target} produced an empty train or test split.")

    model = MajorityClassPredictor().fit(train[["asset_id"]], train[target])
    y_pred = model.predict(test[["asset_id"]])
    metrics = classification_metrics(test[target], y_pred)

    result: dict[str, object] = {
        "target": target,
        "majority_class": str(model.fitted_value_),
        "n_classes": int(train[target].nunique()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        **metrics,
    }

    if log_to_mlflow:
        task = f"T2_{target}"
        run_id = log_classification_run(
            run_name=make_run_name(task, "majority_class"),
            tags=make_standard_tags(
                task=task,
                model_family="baseline",
                model_name="majority_class",
                data_version=data_version,
                split_seed=split_seed,
            ),
            params={
                "target": target,
                "majority_class": result["majority_class"],
                "n_classes": result["n_classes"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "test_size": test_size,
            },
            y_true=test[target],
            y_pred=y_pred,
        )
        result["mlflow_run_id"] = run_id

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run majority-class baselines for categorical attributes."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/master_dataset.csv"),
        help="Processed modeling dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/classification_baselines.csv"),
        help="Where to write the baseline summary CSV.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="Target columns to evaluate.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-version", default="processed-main")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data)

    if not args.no_mlflow:
        setup_mlflow()

    rows: list[dict[str, object]] = []
    for target in args.targets:
        try:
            result = run_target_baseline(
                df,
                target,
                test_size=args.test_size,
                split_seed=args.split_seed,
                data_version=args.data_version,
                log_to_mlflow=not args.no_mlflow,
            )
        except ValueError as exc:
            print(f"Skipping {target}: {exc}")
            continue

        rows.append(result)
        print(
            f"{target}: majority={result['majority_class']!r} "
            f"accuracy={result['accuracy']:.3f} macro_f1={result['macro_f1']:.3f}"
        )

    if not rows:
        raise SystemExit("No baseline runs completed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nWrote {len(rows)} baseline rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

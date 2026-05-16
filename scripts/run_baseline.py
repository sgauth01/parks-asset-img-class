"""Run baseline attribute models and log results to MLflow.

Usage:
    python scripts/run_baseline.py

Outputs:
    results/baseline_classification_results.csv
    results/baseline_classification_cv_folds.csv
    MLflow runs under ./mlruns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.baseline import (  # noqa: E402
    DEFAULT_CLASSIFICATION_TARGETS,
    cross_validate_majority_class,
)


def parse_args() -> argparse.Namespace:
    """Parse command line options for local baseline runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--data-version", default="processed-local")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="parks-asset-img-class")
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Write CSV outputs without logging MLflow runs.",
    )
    return parser.parse_args()


def log_classification_results(
    results: pd.DataFrame,
    artifact_path: Path,
    data_version: str,
    split_seed: int | None,
) -> None:
    """Create one MLflow run per categorical or binned target."""
    import mlflow

    from src.mlflow_utils import make_run_name, make_standard_tags

    for row in results.to_dict(orient="records"):
        attribute = str(row["attribute"])
        with mlflow.start_run(
            run_name=make_run_name(attribute, "majority_class"),
            tags=make_standard_tags(
                task=attribute,
                model_family="baseline",
                model_name="majority_class",
                data_version=data_version,
                split_seed=split_seed,
                extra={"task_type": "classification"},
            ),
        ):
            mlflow.log_param("attribute", attribute)
            mlflow.log_param("strategy", row["strategy"])
            mlflow.log_param("n_folds", int(row["n_folds"]))
            mlflow.log_param("prediction", str(row["prediction"]))
            mlflow.log_metric("n_labels", int(row["n_labels"]))
            mlflow.log_metric(
                "train_majority_share_mean",
                float(row["train_majority_share_mean"]),
            )
            mlflow.log_metric(
                "train_majority_share_std",
                float(row["train_majority_share_std"]),
            )
            mlflow.log_metric("accuracy_mean", float(row["accuracy_mean"]))
            mlflow.log_metric("accuracy_std", float(row["accuracy_std"]))
            mlflow.log_metric("weighted_f1_mean", float(row["weighted_f1_mean"]))
            mlflow.log_metric("weighted_f1_std", float(row["weighted_f1_std"]))
            mlflow.log_metric("macro_f1_mean", float(row["macro_f1_mean"]))
            mlflow.log_metric("macro_f1_std", float(row["macro_f1_std"]))
            mlflow.log_artifact(str(artifact_path))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_path)

    classification_results, fold_results = cross_validate_majority_class(
        df=train_df,
        targets=DEFAULT_CLASSIFICATION_TARGETS,
        n_splits=args.n_splits,
        random_state=args.split_seed,
    )

    classification_path = output_dir / "baseline_classification_results.csv"
    folds_path = output_dir / "baseline_classification_cv_folds.csv"
    classification_results.to_csv(classification_path, index=False)
    fold_results.to_csv(folds_path, index=False)

    mlflow_logged = False
    if not args.skip_mlflow:
        try:
            from src.mlflow_utils import setup_mlflow

            setup_mlflow(experiment_name=args.experiment_name)
            log_classification_results(
                classification_results,
                classification_path,
                data_version=args.data_version,
                split_seed=args.split_seed,
            )
            mlflow_logged = True
        except ModuleNotFoundError as exc:
            if exc.name != "mlflow":
                raise
            print(
                "MLflow is not installed in this Python environment; CSV results "
                "were saved, but MLflow runs were not logged."
            )
            print("Activate the project environment or install mlflow, then rerun:")
            print("  conda activate bcparks_capstone")
            print("  python scripts/run_baseline.py")

    print("\nClassification baselines")
    print(classification_results.to_string(index=False))
    print("\nSaved results to:")
    print(f"  {classification_path}")
    print(f"  {folds_path}")
    if mlflow_logged:
        print("\nView MLflow runs with:")
        print("  mlflow ui --backend-store-uri file:./mlruns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

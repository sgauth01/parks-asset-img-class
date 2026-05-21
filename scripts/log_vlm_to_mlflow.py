"""Log existing Qwen2.5-VL prediction CSVs to DagsHub MLflow.

Reads the three ``qwen2_5_vl_*__new15.csv`` files (predictions copied from
``feat/vlm-onprem-new-split``), scores each model × attribute against the
same per-attribute 85/15 val split used by ``run_dinov3_new_split.py``,
and logs one MLflow run per (model, attribute) under
``model_family=vlm``.

Usage:
    python scripts/log_vlm_to_mlflow.py
    python scripts/log_vlm_to_mlflow.py --no-mlflow   # dry-run, just prints
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_dinov3_vs_vlm import VLM_PIPELINES, score_vlm  # noqa: E402
from scripts.run_dinov3_new_split import ATTRIBUTE_KINDS  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_SEED = 48
DEFAULT_TEST_SIZE = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, default=Path("data/processed/train"))
    p.add_argument("--predictions-dir", type=Path, default=Path("data/predictions/new_split"))
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--data-version", default="per-attribute-85-15")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument(
        "--dagshub-repo",
        default="sgauth01/parks-asset-img-class",
        help="DagsHub repo (owner/name). Pass empty string to use local ./mlruns.",
    )
    return p.parse_args()


def model_slug_from(filename: str) -> str:
    return filename.removesuffix("__new15.csv")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    mlflow = None
    make_run_name = make_standard_tags = setup_mlflow = None
    if not args.no_mlflow:
        try:
            if args.dagshub_repo:
                import dagshub
                owner, name = args.dagshub_repo.split("/", 1)
                dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
            import mlflow as _mlflow
            from src.mlflow_utils import (
                make_run_name as _mrn,
                make_standard_tags as _mst,
                setup_mlflow as _smf,
            )
            _smf()
            mlflow = _mlflow
            make_run_name, make_standard_tags = _mrn, _mst
        except ModuleNotFoundError as exc:
            logger.warning("mlflow/dagshub not installed (%s); dry-run mode", exc.name)

    total_runs = 0
    for filename, label in VLM_PIPELINES:
        path = args.predictions_dir / filename
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        slug = model_slug_from(filename)
        print(f"\n=== {label} ({slug}) ===")
        scores = score_vlm(
            path,
            train_dir=args.train_dir,
            test_size=args.test_size,
            seed=args.split_seed,
        )
        for file_key, kind in ATTRIBUTE_KINDS.items():
            metrics = scores.get(file_key)
            if metrics is None:
                print(f"  {file_key}: (no predictions)")
                continue
            metric_str = (
                f"macro_f1={metrics['macro_f1']:.3f} accuracy={metrics['accuracy']:.3f}"
                if kind == "cls"
                else f"RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f}"
            )
            print(f"  {file_key}: {metric_str} (n_val={metrics['n']})")

            if mlflow is None:
                continue
            task = f"T3_{file_key}"
            tags = make_standard_tags(  # type: ignore[misc]
                task=task,
                model_family="vlm",
                model_name=slug,
                data_version=args.data_version,
                split_seed=args.split_seed,
                extra={
                    "attribute_kind": kind,
                    "split": "per_attribute_85_15",
                    "pipeline_label": label,
                },
            )
            with mlflow.start_run(
                run_name=make_run_name(task, slug),  # type: ignore[misc]
                tags=tags,
            ):
                mlflow.log_param("attribute", file_key)
                mlflow.log_param("model_id", label)
                mlflow.log_param("predictions_file", filename)
                mlflow.log_param("n_val_rows", metrics["n"])
                mlflow.log_param("test_size", args.test_size)
                mlflow.log_param("split_random_state", args.split_seed)
                for k in ("accuracy", "macro_f1", "rmse", "mae"):
                    if k in metrics:
                        mlflow.log_metric(k, float(metrics[k]))
                total_runs += 1

    if mlflow is not None:
        print(f"\nLogged {total_runs} VLM runs to MLflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

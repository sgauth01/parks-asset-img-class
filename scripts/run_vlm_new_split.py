"""Run a VLM against the new suggested per-attribute 85/15 split.

Why a separate script: ``scripts/run_vlm.py`` uses the project's shared
80/20 ``train.csv`` / ``test.csv`` (or its fallback in-memory split) as
the evaluation surface.  The new suggested split is different: each
attribute has its own validation set, derived by
``GroupShuffleSplit(test_size=0.15, random_state=48)`` on the
per-attribute train file.

For a VLM we don't want to re-query the model 12 times per asset, so the
flow here is:

1. Build the union of all per-attribute val ``asset_id`` values.
2. From ``master_dataset.csv`` collect the image rows for those assets.
3. Query the VLM once per asset (multi-image prompt + guided JSON, same
   as ``scripts/run_vlm.py``).
4. Score the VLM predictions independently against every attribute's
   own val rows; emit one row per ``(attribute, image_path)`` pair into
   the same ``data/predictions/new_split/per_attribute_metrics.csv``
   file used by the other new-split pipelines.

Result goes into the same ``data/predictions/new_split/`` directory
(file ``<vlm_slug>__new15.csv``) and the new-split leaderboard at
``reports/leaderboard_new_split.md``.

Usage:
    python scripts/run_vlm_new_split.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model mistralai/Pixtral-12B-2409
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.per_attribute_splits import (  # noqa: E402
    ATTRIBUTE_COLUMNS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_SIZE,
    iter_attribute_splits,
)
from src.data.schema import AttributeKind, load_schema  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    classification_attribute_score,
    numeric_attribute_score,
)
from src.models.vlm_onprem import OpenAICompatibleBackend, predict  # noqa: E402

CLS_KINDS = {
    AttributeKind.CATEGORICAL.value,
    AttributeKind.BOOLEAN.value,
    AttributeKind.ORDINAL_BIN.value,
}

_MISSING_TOKENS = {"", "nan", "none", "null", "tbd", "unknown"}


def _clean_labels(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    mask = out.notna() & ~out.str.lower().isin(_MISSING_TOKENS)
    return out.loc[mask]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="VLM model id served by vLLM.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-images-per-asset", type=int, default=4)
    p.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Cap number of assets queried (smoke testing).",
    )
    p.add_argument("--no-guided-json", action="store_true")
    p.add_argument(
        "--master-csv",
        type=Path,
        default=Path("data/processed/master_dataset.csv"),
        help="Used to map asset_id -> image rows for the VLM prompt.",
    )
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data/predictions/new_split"),
    )
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("data/predictions/new_split/per_attribute_metrics.csv"),
    )
    return p.parse_args()


def _model_slug(model_id: str) -> str:
    return (
        model_id.split("/")[-1]
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def main() -> int:
    args = parse_args()
    schema = load_schema()
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-attribute val sets + the union of val asset_ids
    val_rows_by_attr: dict[str, pd.DataFrame] = {}
    all_val_asset_ids: set[int] = set()
    for attr, _train, val in iter_attribute_splits(
        test_size=args.test_size, random_state=args.split_seed
    ):
        val_rows_by_attr[attr] = val.reset_index(drop=True)
        all_val_asset_ids |= set(val["asset_id"].astype(int).tolist())

    if not val_rows_by_attr:
        raise SystemExit("No per-attribute val rows found.")

    print(
        f"Per-attribute val sets: {len(val_rows_by_attr)} attributes; "
        f"union of val asset_ids = {len(all_val_asset_ids)}."
    )

    # 2. Look up the master rows for those assets so we have image paths
    master = pd.read_csv(args.master_csv)
    if "asset_id" not in master.columns:
        raise SystemExit(f"asset_id column missing from {args.master_csv}")
    test_df = master[master["asset_id"].isin(all_val_asset_ids)].copy().reset_index(drop=True)
    if "file_exists" in test_df.columns:
        test_df = test_df[test_df["file_exists"].astype(bool)].reset_index(drop=True)
    if args.max_assets is not None:
        keep_assets = sorted(test_df["asset_id"].unique())[: args.max_assets]
        test_df = test_df[test_df["asset_id"].isin(keep_assets)].reset_index(drop=True)
    print(
        f"Will query VLM for {test_df['asset_id'].nunique()} assets "
        f"({len(test_df)} image rows total)."
    )

    # 3. Query the VLM
    backend = OpenAICompatibleBackend(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        guided_json=not args.no_guided_json,
    )
    predictions = predict(
        train_df=pd.DataFrame(columns=test_df.columns),  # zero-shot, train ignored
        test_df=test_df,
        schema=schema,
        backend=backend,
        max_images_per_asset=args.max_images_per_asset,
        use_guided_json=not args.no_guided_json,
        max_assets=None,
    )
    if predictions.empty:
        raise SystemExit("VLM produced no predictions.")

    # 4. Persist per-image predictions (joined back to the test_df shape)
    out_csv = args.predictions_dir / f"{_model_slug(args.model)}__new15.csv"
    predictions.to_csv(out_csv, index=False)
    print(f"Wrote per-image predictions to {out_csv}")

    # 5. Score per attribute against per-attribute val rows
    pred_lookup = predictions.set_index("image_path")
    new_rows: list[dict] = []
    for attr_col, val_df in val_rows_by_attr.items():
        attr = schema.attributes[attr_col]
        if attr_col not in pred_lookup.columns:
            print(f"[skip] {attr_col}: VLM did not predict this attribute")
            continue

        # Align predictions to this attribute's val rows by image_path
        val_pred = pred_lookup[attr_col].reindex(val_df["image_path"]).reset_index(drop=True)
        truth = val_df[attr_col].reset_index(drop=True)

        if attr.kind.value in CLS_KINDS:
            t_str = _clean_labels(truth).astype(str)
            p_str = val_pred.astype(str)
            pair = pd.concat(
                [t_str.rename("t"), p_str.rename("p")], axis=1
            ).dropna()
            pair = pair[pair["p"] != "nan"]
            if pair.empty:
                continue
            score = classification_attribute_score(
                attribute=attr_col,
                kind=attr.kind,
                y_true=pair["t"],
                y_pred=pair["p"],
            )
            metric_line = f"macro_f1={score.metrics.get('macro_f1', float('nan')):.3f}"
        else:
            t_num = pd.to_numeric(truth, errors="coerce")
            p_num = pd.to_numeric(val_pred, errors="coerce")
            pair = pd.concat([t_num.rename("t"), p_num.rename("p")], axis=1).dropna()
            if pair.empty:
                continue
            score = numeric_attribute_score(
                attribute=attr_col,
                kind=attr.kind,
                y_true=pair["t"],
                y_pred=pair["p"],
            )
            metric_line = f"rmse={score.metrics.get('rmse', float('nan')):.3f}"

        print(f"{attr_col:34s} -> {metric_line} (n={len(pair)})")

        new_rows.append(
            {
                "pipeline_id": f"vlm_{_model_slug(args.model)}",
                "pipeline_display": f"VLM {args.model.split('/')[-1]}",
                "attribute": attr_col,
                "attribute_display": attr.display_name,
                "kind": attr.kind.value,
                "n_val": int(len(pair)),
                **score.metrics,
            }
        )

    # 6. Append our rows to the shared per-attribute metrics CSV so the
    # leaderboard renderer picks them up alongside the other new-split pipelines.
    if args.metrics_csv.exists():
        existing = pd.read_csv(args.metrics_csv)
        slug = f"vlm_{_model_slug(args.model)}"
        existing = existing[existing["pipeline_id"] != slug]
        out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        out = pd.DataFrame(new_rows)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.metrics_csv, index=False)
    print(f"\nAppended {len(new_rows)} rows to {args.metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render ``reports/leaderboard_new_split.md`` from on-disk metrics.

This script does NOT re-train or re-score anything; it just reads
``data/predictions/new_split/per_attribute_metrics.csv`` and emits the
new-split leaderboard in the same layout as
``scripts/run_per_attribute_eval.py`` produces.  Useful after a VLM run
has appended new rows to the metrics CSV and you want the leaderboard
to reflect them without re-running the lightweight classifiers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.per_attribute_splits import ATTRIBUTE_COLUMNS  # noqa: E402
from src.data.schema import AttributeKind, load_schema  # noqa: E402

CLS_KINDS = {
    AttributeKind.CATEGORICAL.value,
    AttributeKind.BOOLEAN.value,
    AttributeKind.ORDINAL_BIN.value,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("data/predictions/new_split/per_attribute_metrics.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/leaderboard_new_split.md"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.metrics_csv.exists():
        raise SystemExit(f"metrics CSV not found: {args.metrics_csv}")
    df = pd.read_csv(args.metrics_csv)
    schema = load_schema()

    attr_displays = [schema.attributes[c].display_name for c in ATTRIBUTE_COLUMNS]

    lines: list[str] = []
    lines.append("# Leaderboard on the new suggested 85/15 split (asset-grouped, seed=48)")
    lines.append("")
    lines.append(
        "_Each attribute uses its own ``data/processed/train/attr_X_train.csv``, "
        "then ``GroupShuffleSplit(85/15, seed=48)`` by ``asset_id`` for validation._"
    )
    lines.append("")
    lines.append("## Per-attribute scores")
    lines.append("")
    lines.append("| pipeline | " + " | ".join(attr_displays) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in attr_displays) + " |")

    # Order pipelines: dummy baseline first, then lightweights, then VLMs at the end.
    pipelines = (
        df[["pipeline_id", "pipeline_display"]]
        .drop_duplicates()
        .sort_values("pipeline_id")
        .to_records(index=False)
        .tolist()
    )

    # Compute best per attribute (max F1 for cls, min RMSE for num) for bolding.
    best_by_attr: dict[str, float | None] = {}
    for attr in ATTRIBUTE_COLUMNS:
        sub = df[df["attribute"] == attr]
        if sub.empty:
            best_by_attr[attr] = None
            continue
        kind = sub.iloc[0]["kind"]
        if kind in CLS_KINDS:
            v = sub["macro_f1"].dropna()
            best_by_attr[attr] = float(v.max()) if not v.empty else None
        else:
            v = sub["rmse"].dropna()
            best_by_attr[attr] = float(v.min()) if not v.empty else None

    for pid, pdisp in pipelines:
        cells = [pdisp]
        for attr in ATTRIBUTE_COLUMNS:
            row = df[(df["pipeline_id"] == pid) & (df["attribute"] == attr)]
            if row.empty:
                cells.append("—")
                continue
            kind = row.iloc[0]["kind"]
            if kind in CLS_KINDS:
                v = row.iloc[0].get("macro_f1")
            else:
                v = row.iloc[0].get("rmse")
            if v is None or pd.isna(v):
                cells.append("—")
            else:
                best = best_by_attr.get(attr)
                tag = "**" if best is not None and float(v) == float(best) else ""
                cells.append(f"{tag}{float(v):.3f}{tag}")
        lines.append("| " + " | ".join(cells) + " |")

    # Cross-attribute aggregates.
    lines.append("")
    lines.append("## Cross-attribute aggregates")
    lines.append("")
    lines.append(
        "| pipeline | unweighted macro-F1 (cls) | "
        "unweighted RMSE (numeric) | unweighted MAE (numeric) |"
    )
    lines.append("| --- | --- | --- | --- |")
    for pid, pdisp in pipelines:
        sub = df[df["pipeline_id"] == pid]
        cls = sub[sub["kind"].isin(CLS_KINDS)]
        num = sub[~sub["kind"].isin(CLS_KINDS)]
        cls_f1 = cls["macro_f1"].dropna() if "macro_f1" in cls.columns else pd.Series([], dtype=float)
        num_rmse = num["rmse"].dropna() if "rmse" in num.columns else pd.Series([], dtype=float)
        num_mae = num["mae"].dropna() if "mae" in num.columns else pd.Series([], dtype=float)
        lines.append(
            "| "
            + pdisp
            + " | "
            + (f"{cls_f1.mean():.3f}" if not cls_f1.empty else "—")
            + " | "
            + (f"{num_rmse.mean():.3f}" if not num_rmse.empty else "—")
            + " | "
            + (f"{num_mae.mean():.3f}" if not num_mae.empty else "—")
            + " |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

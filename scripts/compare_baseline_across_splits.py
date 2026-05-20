"""Head-to-head baseline scores: project 80/20 split vs the new suggested 85/15 split.

Writes ``reports/baseline_split_comparison.md`` so the team can drop the
numbers straight into a follow-up message and know exactly what
baseline a real model has to beat.

The baseline is the same on both splits — majority class for
classification / boolean / ordinal-bin attributes, median for numeric.
The only thing that changes between the two columns is *the
training/test partition*:

- **Project split**: shared 80/20 ``data/processed/train.csv`` /
  ``test.csv`` (asset-grouped, seed=42).  This drives every row of
  ``reports/leaderboard.md`` and the PDF report.
- **New suggested split**: per-attribute 85/15 inside each
  ``data/processed/train/attr_X_train.csv`` via
  ``GroupShuffleSplit(test_size=0.15, random_state=48)`` keyed by
  ``asset_id``.

Each attribute therefore has a different validation set across the two
splits, so the baseline numbers will differ even though the model is
identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import AttributeKind, load_schema  # noqa: E402


CLS_KINDS = {
    AttributeKind.CATEGORICAL.value,
    AttributeKind.BOOLEAN.value,
    AttributeKind.ORDINAL_BIN.value,
}


def _project_baseline_rows() -> pd.DataFrame:
    """Baseline rows from the project 80/20 (test) per-attribute table."""
    path = REPO_ROOT / "reports" / "report_tables" / "per_attribute.csv"
    df = pd.read_csv(path)
    return df[df["pipeline"] == "baseline"].copy()


def _new_split_baseline_rows() -> pd.DataFrame:
    """Baseline rows from the per-attribute new-suggested-split metrics CSV."""
    path = REPO_ROOT / "data" / "predictions" / "new_split" / "per_attribute_metrics.csv"
    df = pd.read_csv(path)
    return df[df["pipeline_id"] == "baseline"].copy()


def main() -> int:
    schema = load_schema()
    proj = _project_baseline_rows().set_index("attribute")
    new = _new_split_baseline_rows().set_index("attribute")

    out_path = REPO_ROOT / "reports" / "baseline_split_comparison.md"

    lines: list[str] = []
    lines.append("# Baseline (Dummy classifier / Median) — split-to-split comparison")
    lines.append("")
    lines.append(
        "Same model, two evaluation surfaces.  Both runs are on the same "
        "`master_dataset.csv`; only the train/val partition differs."
    )
    lines.append("")
    lines.append(
        "- **Project split**: shared 80/20 asset-grouped split with "
        "`random_state=42` (see `data/processed/train.csv` / `test.csv`)."
    )
    lines.append(
        "- **New suggested split (per-attribute)**: each attribute uses its "
        "own `data/processed/train/attr_X_train.csv`, then "
        "`GroupShuffleSplit(test_size=0.15, random_state=48)` by "
        "`asset_id` for validation."
    )
    lines.append("")
    lines.append("Cls/bool attributes report **macro-F1** (higher is better); ")
    lines.append("numeric/count attributes report **RMSE** and **MAE** (lower is better).")
    lines.append("")
    lines.append(
        "| Attribute | Kind | Project (80/20) | New suggested (85/15) | Δ | Notes |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")

    proj_metrics: dict[str, float] = {}
    new_metrics: dict[str, float] = {}

    for attr_col in schema.attribute_columns():
        attr = schema.attributes[attr_col]
        kind = attr.kind.value
        proj_row = proj.loc[attr_col] if attr_col in proj.index else None
        new_row = new.loc[attr_col] if attr_col in new.index else None
        if proj_row is None and new_row is None:
            continue

        if kind in CLS_KINDS:
            p = float(proj_row["macro_f1"]) if proj_row is not None else None
            n = float(new_row["macro_f1"]) if new_row is not None else None
            metric = "macro-F1"
            fmt = lambda v: "—" if v is None else f"{v:.3f}"
            higher_better = True
        else:
            p = float(proj_row["rmse"]) if proj_row is not None else None
            n = float(new_row["rmse"]) if new_row is not None else None
            metric = "RMSE"
            fmt = lambda v: "—" if v is None else f"{v:.3f}"
            higher_better = False

        if p is not None and n is not None:
            delta = n - p
            if (delta > 0 and higher_better) or (delta < 0 and not higher_better):
                arrow = "↑ better"
            elif delta == 0:
                arrow = "="
            else:
                arrow = "↓ worse"
            delta_str = f"{delta:+.3f} ({arrow})"
        else:
            delta_str = "—"

        notes = ""
        if proj_row is not None and new_row is not None and kind not in CLS_KINDS:
            # Numeric attributes — flag big swings driven by outliers.
            if p is not None and n is not None:
                if n > p * 1.5 or p > n * 1.5:
                    notes = "val set composition differs heavily"
        if proj_row is not None and new_row is not None:
            p_n = int(proj_row.get("n", 0))
            n_n = int(new_row.get("n_val", 0))
            notes = (notes + (" · " if notes else "") + f"n_test_proj={p_n}, n_val_new={n_n}")

        lines.append(
            f"| {attr.display_name} ({metric}) | {kind} | {fmt(p)} | {fmt(n)} | {delta_str} | {notes} |"
        )

        if kind in CLS_KINDS and p is not None and n is not None:
            proj_metrics[attr_col] = p
            new_metrics[attr_col] = n

    # Cross-attribute aggregate of macro-F1 (matches what leaderboards show)
    proj_macro = sum(proj_metrics.values()) / len(proj_metrics) if proj_metrics else float("nan")
    new_macro = sum(new_metrics.values()) / len(new_metrics) if new_metrics else float("nan")
    lines.append("")
    lines.append("## Cross-attribute aggregate (classification only)")
    lines.append("")
    lines.append(
        f"- Project split (80/20)         unweighted macro-F1 over cls attrs: **{proj_macro:.3f}**"
    )
    lines.append(
        f"- New suggested split (85/15)   unweighted macro-F1 over cls attrs: **{new_macro:.3f}**"
    )
    lines.append(f"- Δ = {new_macro - proj_macro:+.3f}")
    lines.append("")
    lines.append(
        "_Interpretation: the dummy classifier hasn't actually changed; "
        "the small per-attribute differences come from each attribute's "
        "validation set being a different subset of the labelled rows. "
        "Large per-attribute swings on numeric RMSE (e.g. Length, Width) "
        "are driven by the new split's per-attribute training file "
        "containing different outliers than the project test file (e.g. "
        "the 2,600 m boardwalk and the 0.4 m boardwalk are placed in "
        "train or val depending on the asset_id partition)._"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path.read_text())
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

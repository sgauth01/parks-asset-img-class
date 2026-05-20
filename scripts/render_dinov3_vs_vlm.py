"""Score DINOv3 + 3 Qwen VLM predictions vs the per-attribute 85/15 val split,
emit reports/dinov3_vs_vlm_new_split.md.

Reads:
- data/predictions/new_split/{dinov3_model}__{head}_{numeric_head}__new15.csv
  (long format: image_path, asset_id, attribute, y_true, y_pred — from
  scripts/run_dinov3_new_split.py)
- data/predictions/new_split/qwen2_5_vl_*__new15.csv
  (wide format: image_path, attr_X, attr_X__confidence — from VLM PR runner)
- data/processed/train/attr_*_train.csv (ground truth + asset_id for split)

Writes:
- reports/dinov3_vs_vlm_new_split.md

Usage:
    python scripts/render_dinov3_vs_vlm.py
    python scripts/render_dinov3_vs_vlm.py --dinov3-model facebook/dinov2-large
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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

from scripts.run_dinov3_new_split import (  # noqa: E402
    ATTRIBUTE_KINDS,
    MISSING_LABELS,
    asset_grouped_split,
    resolve_column,
)

DEFAULT_DINOV3_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_DINOV3_HEAD = "logistic"
DEFAULT_DINOV3_NUMERIC_HEAD = "ridge"

# Display labels per attribute, matching the existing VLM report.
ATTR_DISPLAY: dict[str, str] = {
    "attr_abutment_material": "Abutment Material",
    "attr_bridge_type": "Bridge Type",
    "attr_decking_material": "Decking Material",
    "attr_fall_height": "Fall Height (m)",
    "attr_has_edge_guard": "Has Edge Guard",
    "attr_has_pedestrian_railing": "Has Pedestrian Railing",
    "attr_length": "Length (m)",
    "attr_material_frame_tank_body": "Material (Frame, Tank, Body)",
    "attr_number_of_steps": "Number of Steps",
    "attr_structure_material": "Structure Material",
    "attr_structure_position": "Structure Position",
    "attr_width": "Width (m)",
}

CLS_ATTRS = [k for k, v in ATTRIBUTE_KINDS.items() if v == "cls"]
NUM_ATTRS = [k for k, v in ATTRIBUTE_KINDS.items() if v == "num"]

VLM_PIPELINES = [
    ("qwen2_5_vl_7b_instruct__new15.csv", "Qwen-7B"),
    ("qwen2_5_vl_32b_instruct_awq__new15.csv", "Qwen-32B-AWQ"),
    ("qwen2_5_vl_72b_instruct_awq__new15.csv", "Qwen-72B-AWQ"),
]


def _slug_for_model(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_").lower()


def _val_df_for(file_key: str, train_dir: Path, *, test_size: float, seed: int) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(train_dir / f"{file_key}_train.csv")
    col = resolve_column(df, file_key)
    _, val_df = asset_grouped_split(df, test_size=test_size, random_state=seed)
    return val_df, col


def _clean_class_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.where(out.notna() & ~out.str.lower().isin(MISSING_LABELS))


def _score_cls(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float] | None:
    yt = _clean_class_series(y_true)
    yp = _clean_class_series(y_pred)
    keep = yt.notna() & yp.notna()
    if keep.sum() == 0:
        return None
    yt_v = yt[keep].tolist()
    yp_v = yp[keep].tolist()
    return {
        "n": int(keep.sum()),
        "accuracy": float(accuracy_score(yt_v, yp_v)),
        "macro_f1": float(f1_score(yt_v, yp_v, average="macro", zero_division=0)),
    }


def _score_num(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float] | None:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    keep = yt.notna() & yp.notna()
    if keep.sum() == 0:
        return None
    yt_v = yt[keep].values
    yp_v = yp[keep].values
    return {
        "n": int(keep.sum()),
        "mae": float(mean_absolute_error(yt_v, yp_v)),
        "rmse": float(math.sqrt(mean_squared_error(yt_v, yp_v))),
    }


def score_dinov3(
    pred_path: Path, *, train_dir: Path, test_size: float, seed: int
) -> dict[str, dict[str, float]]:
    """DINOv3 predictions are long-format with y_true / y_pred columns."""
    if not pred_path.exists():
        raise FileNotFoundError(f"DINOv3 predictions not found at {pred_path}")
    preds = pd.read_csv(pred_path)
    out: dict[str, dict[str, float]] = {}
    for file_key, kind in ATTRIBUTE_KINDS.items():
        subset = preds[preds["attribute"] == file_key]
        if subset.empty:
            continue
        score = _score_cls(subset["y_true"], subset["y_pred"]) if kind == "cls" else _score_num(subset["y_true"], subset["y_pred"])
        if score is not None:
            out[file_key] = score
    return out


def score_vlm(
    pred_path: Path, *, train_dir: Path, test_size: float, seed: int
) -> dict[str, dict[str, float]]:
    """VLM predictions are wide-format with one column per attribute."""
    if not pred_path.exists():
        raise FileNotFoundError(f"VLM predictions not found at {pred_path}")
    preds = pd.read_csv(pred_path)
    out: dict[str, dict[str, float]] = {}
    for file_key, kind in ATTRIBUTE_KINDS.items():
        val_df, col = _val_df_for(file_key, train_dir, test_size=test_size, seed=seed)
        if col not in preds.columns and file_key not in preds.columns:
            continue
        # VLM CSV uses the original (possibly comma-bearing) attribute column name
        pred_col = col if col in preds.columns else file_key
        joined = val_df[["image_path", "asset_id", col]].merge(
            preds[["image_path", pred_col]].rename(columns={pred_col: f"{col}__pred"}),
            on="image_path",
            how="left",
        )
        if joined.empty:
            continue
        score = (
            _score_cls(joined[col], joined[f"{col}__pred"])
            if kind == "cls"
            else _score_num(joined[col], joined[f"{col}__pred"])
        )
        if score is not None:
            out[file_key] = score
    return out


def _fmt(v: float | None, *, digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _best_idx(values: list[float | None], *, higher_is_better: bool) -> int | None:
    finite = [(i, v) for i, v in enumerate(values) if v is not None]
    if not finite:
        return None
    if higher_is_better:
        return max(finite, key=lambda iv: iv[1])[0]
    return min(finite, key=lambda iv: iv[1])[0]


def render_table(
    *,
    attrs: list[str],
    metric: str,
    higher_is_better: bool,
    pipelines: list[str],
    scores: list[dict[str, dict[str, float]]],
    digits: int = 3,
) -> str:
    header = "| Attribute | " + " | ".join(pipelines) + " |"
    sep = "|---|" + "---:|" * len(pipelines)
    rows = [header, sep]
    means: list[list[float]] = [[] for _ in pipelines]

    for attr in attrs:
        vals = [s.get(attr, {}).get(metric) for s in scores]
        for i, v in enumerate(vals):
            if v is not None and not math.isnan(v):
                means[i].append(v)
        bi = _best_idx(vals, higher_is_better=higher_is_better)
        cells = [_fmt(v, digits=digits) for v in vals]
        if bi is not None:
            cells[bi] = f"**{cells[bi]}**"
        rows.append(f"| {ATTR_DISPLAY[attr]} | " + " | ".join(cells) + " |")

    mean_vals: list[float | None] = [
        (sum(m) / len(m)) if m else None for m in means
    ]
    bi_mean = _best_idx(mean_vals, higher_is_better=higher_is_better)
    mean_cells = [_fmt(v, digits=digits) for v in mean_vals]
    if bi_mean is not None:
        mean_cells[bi_mean] = f"**{mean_cells[bi_mean]}**"
    rows.append("| **Mean** | " + " | ".join(mean_cells) + " |")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dinov3-model", default=DEFAULT_DINOV3_MODEL)
    p.add_argument("--dinov3-head", default=DEFAULT_DINOV3_HEAD)
    p.add_argument("--dinov3-numeric-head", default=DEFAULT_DINOV3_NUMERIC_HEAD)
    p.add_argument("--dinov3-label", default="DINOv3-L")
    p.add_argument("--train-dir", type=Path, default=Path("data/processed/train"))
    p.add_argument("--predictions-dir", type=Path, default=Path("data/predictions/new_split"))
    p.add_argument("--output", type=Path, default=Path("reports/dinov3_vs_vlm_new_split.md"))
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=48)
    args = p.parse_args()

    dinov3_slug = _slug_for_model(args.dinov3_model)
    dinov3_path = args.predictions_dir / f"{dinov3_slug}__{args.dinov3_head}_{args.dinov3_numeric_head}__new15.csv"
    dinov3_scores = score_dinov3(
        dinov3_path,
        train_dir=args.train_dir,
        test_size=args.test_size,
        seed=args.split_seed,
    )
    vlm_scores: list[dict[str, dict[str, float]]] = []
    vlm_labels: list[str] = []
    for fname, label in VLM_PIPELINES:
        path = args.predictions_dir / fname
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        vlm_scores.append(
            score_vlm(path, train_dir=args.train_dir, test_size=args.test_size, seed=args.split_seed)
        )
        vlm_labels.append(label)

    pipelines = [args.dinov3_label] + vlm_labels
    all_scores = [dinov3_scores] + vlm_scores

    lines = [
        "# DINOv3 vs VLM — new 85/15 split",
        "",
        f"All pipelines evaluated on the same `data/processed/train/attr_*_train.csv` "
        f"files with `GroupShuffleSplit(test_size={args.test_size}, "
        f"random_state={args.split_seed})` keyed on `asset_id`.",
        "",
        f"- **{args.dinov3_label}**: `{args.dinov3_model}` features + "
        f"`{args.dinov3_head}` (classification) / `{args.dinov3_numeric_head}` (numeric).",
    ]
    for label in vlm_labels:
        lines.append(f"- **{label}**: Qwen2.5-VL on-prem via vLLM (zero-shot).")

    lines += [
        "",
        "---",
        "",
        "## Classification & boolean — macro-F1 (higher is better)",
        "",
        render_table(
            attrs=CLS_ATTRS,
            metric="macro_f1",
            higher_is_better=True,
            pipelines=pipelines,
            scores=all_scores,
        ),
        "",
        "---",
        "",
        "## Numeric & count — RMSE (lower is better)",
        "",
        render_table(
            attrs=NUM_ATTRS,
            metric="rmse",
            higher_is_better=False,
            pipelines=pipelines,
            scores=all_scores,
        ),
        "",
        "---",
        "",
        "## Numeric & count — MAE (lower is better)",
        "",
        render_table(
            attrs=NUM_ATTRS,
            metric="mae",
            higher_is_better=False,
            pipelines=pipelines,
            scores=all_scores,
        ),
        "",
        "---",
        "",
        "## Headline",
        "",
        _headline_table(pipelines, all_scores),
        "",
        "---",
        "",
        "## Notes",
        "",
        f"- Per issue #41 — DINOv3 trains one head per attribute on the cached "
        f"`{args.dinov3_model}` features and predicts on the held-out 15% (asset-grouped).",
        "- VLM rows reuse predictions from `feat/vlm-onprem-new-split` (no rerun).",
        "- `Qwen-7B` returned numeric values for only `Number of Steps` (n=2 val rows); "
        "its `0.000` RMSE / MAE is a 2-row lucky guess, not a real signal. Its "
        "numeric mean column is therefore a 1-attribute average and should be read "
        "alongside its missing entries above.",
        "- DINOv3 `Length (m)` RMSE is dominated by the same 2,600 m boardwalk outlier "
        "that lives in the training set but not val — ridge regression on global "
        "features cannot identify it. The VLMs read the image and answer directly.",
        f"- Source CSVs: `{dinov3_path.name}` + the three Qwen `*__new15.csv` files in "
        "`data/predictions/new_split/`.",
        f"- Regenerate with `python scripts/render_dinov3_vs_vlm.py`.",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")
    return 0


def _headline_table(pipelines: list[str], scores: list[dict[str, dict[str, float]]]) -> str:
    header = "| | " + " | ".join(pipelines) + " |"
    sep = "|---|" + "---:|" * len(pipelines)

    def mean_metric(s: dict[str, dict[str, float]], attrs: list[str], metric: str) -> float | None:
        vals = [s[a][metric] for a in attrs if a in s and metric in s[a] and not math.isnan(s[a][metric])]
        return sum(vals) / len(vals) if vals else None

    macro = [mean_metric(s, CLS_ATTRS, "macro_f1") for s in scores]
    rmse = [mean_metric(s, NUM_ATTRS, "rmse") for s in scores]
    mae = [mean_metric(s, NUM_ATTRS, "mae") for s in scores]

    def row(label: str, vals: list[float | None], *, higher_is_better: bool, digits: int = 3) -> str:
        bi = _best_idx(vals, higher_is_better=higher_is_better)
        cells = [_fmt(v, digits=digits) for v in vals]
        if bi is not None:
            cells[bi] = f"**{cells[bi]}**"
        return f"| {label} | " + " | ".join(cells) + " |"

    return "\n".join([
        header,
        sep,
        row("macro-F1 (cls)", macro, higher_is_better=True),
        row("RMSE (numeric)", rmse, higher_is_better=False, digits=3),
        row("MAE (numeric)", mae, higher_is_better=False, digits=3),
    ])


if __name__ == "__main__":
    raise SystemExit(main())

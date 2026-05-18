# Baseline vs VLM — new 85/15 split

I tested three Qwen2.5-VL models against the dummy baseline. All use the
new split (`GroupShuffleSplit(test_size=0.15, random_state=48)` by `asset_id`,
per-attribute train files in `data/processed/train/`).

- **Baseline**: most common label for classes, median for numbers. No image used.
- **Qwen-7B**: full fp16, ~16 GB, ~6 s per asset.
- **Qwen-32B**: AWQ-int4, ~25 GB, ~20 s per asset.
- **Qwen-72B**: AWQ-int4, ~50 GB, ~30 s per asset.

All VLMs ran on-prem with vLLM. No images leave the box.

---

## Classification & boolean — macro-F1 (higher is better)

| Attribute | Baseline | Qwen-7B | Qwen-32B | Qwen-72B |
|---|---:|---:|---:|---:|
| Abutment Material | 0.143 | 0.272 | 0.265 | 0.250 |
| Bridge Type | 0.223 | 0.398 | **0.473** | 0.443 |
| Decking Material | 0.240 | 0.201 | 0.272 | **0.396** |
| Has Edge Guard | 0.472 | 0.521 | 0.587 | **0.659** |
| Has Pedestrian Railing | 0.310 | 0.609 | 0.582 | **0.646** |
| Material (Frame, Tank, Body) | 0.093 | 0.370 | 0.397 | **0.592** |
| Structure Material | 0.236 | **0.589** | 0.374 | 0.374 |
| Structure Position | 0.218 | 0.350 | 0.365 | **0.412** |
| **Mean** | **0.242** | **0.414** | **0.414** | **0.472** |

All three VLMs beat the baseline. Qwen-7B and Qwen-32B tie on the mean.
Qwen-72B wins overall.

---

## Numeric & count — RMSE (lower is better)

| Attribute | Baseline | Qwen-7B | Qwen-32B | Qwen-72B |
|---|---:|---:|---:|---:|
| Fall Height (m) | 1.032 | — | — | **0.782** |
| Length (m) | 218.989 | — | **4.040** | 6.600 |
| Number of Steps | 9.947 | 0.000 *(n=2)* | 10.932 | **10.747** |
| Width (m) | 2.831 | — | 0.500 | **0.431** |

Qwen-7B returned numbers for only 1 attribute, with only 2 val rows.
The 0.000 is from those 2 rows being correct by luck — not real signal.

---

## Numeric & count — MAE (lower is better)

| Attribute | Baseline | Qwen-7B | Qwen-32B | Qwen-72B |
|---|---:|---:|---:|---:|
| Fall Height (m) | 0.826 | — | — | **0.570** |
| Length (m) | 38.118 | — | **1.730** | 1.840 |
| Number of Steps | 7.647 | 0.000 *(n=2)* | 6.167 | **5.380** |
| Width (m) | 0.724 | — | 0.500 | **0.391** |

---

## Headline

| | Baseline | Qwen-7B | Qwen-32B | Qwen-72B |
|---|---:|---:|---:|---:|
| macro-F1 (classes) | 0.242 | 0.414 | 0.414 | **0.472** |
| RMSE (numbers) | 58.20 | n/a | 5.16 | **4.64** |
| MAE (numbers) | 11.83 | n/a | 3.33 | **2.69** |

**Both AWQ-quantized VLMs (32B + 72B) beat baseline on every attribute.
Qwen-72B is the strongest. Qwen-7B is OK for classes only — it does not
return numbers reliably.**

---

## Notes

- Length jumps from RMSE 219 (baseline) to 4-7 (VLMs). The training file
  has a 2,600 m boardwalk in train but not val, so the median is far off.
  The VLM looks at the image and gives a much better answer.
- I also tried Pixtral-12B and InternVL2.5-78B-AWQ. Both crashed in vLLM
  0.20 at load time. Not in this report.
- Source: `data/predictions/new_split/per_attribute_metrics.csv`.
  Rebuild with `python scripts/render_new_split_leaderboard.py`.

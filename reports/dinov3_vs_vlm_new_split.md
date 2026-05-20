# DINOv3 vs VLM — new 85/15 split

All pipelines evaluated on the same `data/processed/train/attr_*_train.csv` files with `GroupShuffleSplit(test_size=0.15, random_state=48)` keyed on `asset_id`.

- **DINOv3-L**: `facebook/dinov3-vitl16-pretrain-lvd1689m` features + `logistic` (classification) / `ridge` (numeric).
- **Qwen-7B**: Qwen2.5-VL on-prem via vLLM (zero-shot).
- **Qwen-32B-AWQ**: Qwen2.5-VL on-prem via vLLM (zero-shot).
- **Qwen-72B-AWQ**: Qwen2.5-VL on-prem via vLLM (zero-shot).

---

## Classification & boolean — macro-F1 (higher is better)

| Attribute | DINOv3-L | Qwen-7B | Qwen-32B-AWQ | Qwen-72B-AWQ |
|---|---:|---:|---:|---:|
| Abutment Material | **0.374** | 0.272 | 0.265 | 0.250 |
| Bridge Type | 0.394 | 0.398 | **0.473** | 0.443 |
| Decking Material | 0.387 | 0.201 | 0.272 | **0.396** |
| Has Edge Guard | **0.695** | 0.521 | 0.587 | 0.659 |
| Has Pedestrian Railing | 0.600 | 0.609 | 0.582 | **0.646** |
| Material (Frame, Tank, Body) | 0.469 | 0.370 | 0.397 | **0.592** |
| Structure Material | 0.494 | **0.589** | 0.374 | 0.374 |
| Structure Position | **0.896** | 0.350 | 0.365 | 0.412 |
| **Mean** | **0.539** | 0.414 | 0.414 | 0.472 |

---

## Numeric & count — RMSE (lower is better)

| Attribute | DINOv3-L | Qwen-7B | Qwen-32B-AWQ | Qwen-72B-AWQ |
|---|---:|---:|---:|---:|
| Fall Height (m) | 1.341 | — | — | **0.782** |
| Length (m) | 219.152 | — | **4.040** | 6.600 |
| Number of Steps | 10.227 | **0.000** | 10.932 | 10.747 |
| Width (m) | 1.878 | — | 0.500 | **0.431** |
| **Mean** | 58.150 | **0.000** | 5.157 | 4.640 |

---

## Numeric & count — MAE (lower is better)

| Attribute | DINOv3-L | Qwen-7B | Qwen-32B-AWQ | Qwen-72B-AWQ |
|---|---:|---:|---:|---:|
| Fall Height (m) | 1.135 | — | — | **0.689** |
| Length (m) | 52.000 | — | **3.320** | 3.894 |
| Number of Steps | 9.498 | **0.000** | 6.167 | 5.833 |
| Width (m) | 0.678 | — | 0.500 | **0.350** |
| **Mean** | 15.828 | **0.000** | 3.329 | 2.692 |

---

## Headline

| | DINOv3-L | Qwen-7B | Qwen-32B-AWQ | Qwen-72B-AWQ |
|---|---:|---:|---:|---:|
| macro-F1 (cls) | **0.539** | 0.414 | 0.414 | 0.472 |
| RMSE (numeric) | 58.150 | **0.000** | 5.157 | 4.640 |
| MAE (numeric) | 15.828 | **0.000** | 3.329 | 2.692 |

---

## Notes

- Per issue #41 — DINOv3 trains one head per attribute on the cached `facebook/dinov3-vitl16-pretrain-lvd1689m` features and predicts on the held-out 15% (asset-grouped).
- VLM rows reuse predictions from `feat/vlm-onprem-new-split` (no rerun).
- `Qwen-7B` returned numeric values for only `Number of Steps` (n=2 val rows); its `0.000` RMSE / MAE is a 2-row lucky guess, not a real signal. Its numeric mean column is therefore a 1-attribute average and should be read alongside its missing entries above.
- DINOv3 `Length (m)` RMSE is dominated by the same 2,600 m boardwalk outlier that lives in the training set but not val — ridge regression on global features cannot identify it. The VLMs read the image and answer directly.
- Source CSVs: `facebook_dinov3_vitl16_pretrain_lvd1689m__logistic_ridge__new15.csv` + the three Qwen `*__new15.csv` files in `data/predictions/new_split/`.
- Regenerate with `python scripts/render_dinov3_vs_vlm.py`.

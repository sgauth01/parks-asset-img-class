# Baseline (Dummy classifier / Median) — split-to-split comparison

Same model, two evaluation surfaces.  Both runs are on the same `master_dataset.csv`; only the train/val partition differs.

- **Project split**: shared 80/20 asset-grouped split with `random_state=42` (see `data/processed/train.csv` / `test.csv`).
- **New suggested split (per-attribute)**: each attribute uses its own `data/processed/train/attr_X_train.csv`, then `GroupShuffleSplit(test_size=0.15, random_state=48)` by `asset_id` for validation.

Cls/bool attributes report **macro-F1** (higher is better); 
numeric/count attributes report **RMSE** and **MAE** (lower is better).

| Attribute | Kind | Project (80/20) | New suggested (85/15) | Δ | Notes |
| --- | --- | --- | --- | --- | --- |
| Abutment Material (macro-F1) | categorical | 0.139 | 0.143 | +0.004 (↑ better) | n_test_proj=242, n_val_new=172 |
| Bridge Type (macro-F1) | categorical | 0.178 | 0.223 | +0.044 (↑ better) | n_test_proj=271, n_val_new=172 |
| Decking Material (macro-F1) | categorical | 0.193 | 0.240 | +0.046 (↑ better) | n_test_proj=680, n_val_new=448 |
| Fall Height (m) (RMSE) | numeric | 4.260 | 1.032 | -3.227 (↑ better) | val set composition differs heavily · n_test_proj=62, n_val_new=31 |
| Has Edge Guard (macro-F1) | boolean | 0.490 | 0.472 | -0.018 (↓ worse) | n_test_proj=384, n_val_new=397 |
| Has Pedestrian Railing (macro-F1) | categorical | 0.309 | 0.310 | +0.001 (↑ better) | n_test_proj=832, n_val_new=558 |
| Length (m) (RMSE) | numeric | 82.151 | 218.989 | +136.839 (↓ worse) | val set composition differs heavily · n_test_proj=619, n_val_new=401 |
| Material (Frame, Tank, Body) (macro-F1) | categorical | 0.085 | 0.093 | +0.008 (↑ better) | n_test_proj=205, n_val_new=212 |
| Number of Steps (RMSE) | count | 10.775 | 9.947 | -0.828 (↑ better) | n_test_proj=10, n_val_new=17 |
| Structure Material (macro-F1) | categorical | 0.214 | 0.236 | +0.022 (↑ better) | n_test_proj=295, n_val_new=396 |
| Structure Position (macro-F1) | categorical | 0.278 | 0.218 | -0.060 (↓ worse) | n_test_proj=205, n_val_new=150 |
| Width (m) (RMSE) | numeric | 1.411 | 2.831 | +1.420 (↓ worse) | val set composition differs heavily · n_test_proj=632, n_val_new=432 |

## Cross-attribute aggregate (classification only)

- Project split (80/20)         unweighted macro-F1 over cls attrs: **0.236**
- New suggested split (85/15)   unweighted macro-F1 over cls attrs: **0.242**
- Δ = +0.006

_Interpretation: the dummy classifier hasn't actually changed; the small per-attribute differences come from each attribute's validation set being a different subset of the labelled rows. Large per-attribute swings on numeric RMSE (e.g. Length, Width) are driven by the new split's per-attribute training file containing different outliers than the project test file (e.g. the 2,600 m boardwalk and the 0.4 m boardwalk are placed in train or val depending on the asset_id partition)._
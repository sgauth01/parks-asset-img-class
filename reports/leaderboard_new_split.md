# Leaderboard on the new suggested 85/15 split (asset-grouped, seed=48)

_Each attribute uses its own ``data/processed/train/attr_X_train.csv``, then ``GroupShuffleSplit(85/15, seed=48)`` by ``asset_id`` for validation._

## Per-attribute scores

| pipeline | Abutment Material | Bridge Type | Decking Material | Fall Height (m) | Has Edge Guard | Has Pedestrian Railing | Length (m) | Material (Frame, Tank, Body) | Number of Steps | Structure Material | Structure Position | Width (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (majority / median) | 0.143 | 0.223 | 0.240 | 1.032 | 0.472 | 0.310 | 218.989 | 0.093 | 9.947 | 0.236 | 0.218 | 2.831 |
| VLM Qwen2.5-VL-32B-Instruct-AWQ | 0.265 | **0.473** | 0.272 | — | 0.587 | 0.582 | **4.040** | 0.397 | 10.932 | 0.374 | 0.365 | 0.500 |
| VLM Qwen2.5-VL-72B-Instruct-AWQ | 0.250 | 0.443 | **0.396** | **0.782** | **0.659** | **0.646** | 6.600 | **0.592** | 10.747 | 0.374 | **0.412** | **0.431** |
| VLM Qwen2.5-VL-7B-Instruct | **0.272** | 0.398 | 0.201 | — | 0.521 | 0.609 | — | 0.370 | **0.000** | **0.589** | 0.350 | — |

## Cross-attribute aggregates

| pipeline | unweighted macro-F1 (cls) | unweighted RMSE (numeric) | unweighted MAE (numeric) |
| --- | --- | --- | --- |
| Baseline (majority / median) | 0.242 | 58.200 | 11.829 |
| VLM Qwen2.5-VL-32B-Instruct-AWQ | 0.414 | 5.157 | 3.329 |
| VLM Qwen2.5-VL-72B-Instruct-AWQ | 0.472 | 4.640 | 2.692 |
| VLM Qwen2.5-VL-7B-Instruct | 0.414 | 0.000 | 0.000 |

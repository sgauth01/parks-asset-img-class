# Leaderboard on the new suggested 85/15 split (asset-grouped, seed=48)

_Each attribute uses its own ``data/processed/train/attr_X_train.csv``, then GroupShuffleSplit(85/15, seed=48) by ``asset_id`` for validation._

## Per-attribute scores

| pipeline | Abutment Material | Bridge Type | Decking Material | Fall Height (m) | Has Edge Guard | Has Pedestrian Railing | Length (m) | Material (Frame, Tank, Body) | Number of Steps | Structure Material | Structure Position | Width (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (majority / median) | 0.143 | 0.223 | 0.240 | 1.032 | 0.472 | 0.310 | 218.989 | 0.093 | 9.947 | 0.236 | 0.218 | 2.831 |

## Cross-attribute aggregates

| pipeline | unweighted macro-F1 (cls) | unweighted RMSE (numeric) | unweighted MAE (numeric) |
| --- | --- | --- | --- |
| Baseline (majority / median) | 0.242 | 58.200 | 11.829 |

# Leaderboard on the new suggested 85/15 split (asset-grouped, seed=48)

_Each attribute uses its own ``data/processed/train/attr_X_train.csv``, then GroupShuffleSplit(85/15, seed=48) by ``asset_id`` for validation._

## Per-attribute scores

| pipeline | Abutment Material | Bridge Type | Decking Material | Fall Height (m) | Has Edge Guard | Has Pedestrian Railing | Length (m) | Material (Frame, Tank, Body) | Number of Steps | Structure Material | Structure Position | Width (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (majority / median) | 0.143 | 0.223 | 0.240 | 1.032 | **0.472** | 0.310 | 218.989 | 0.093 | 9.947 | 0.236 | 0.218 | 2.831 |
| DINOv2-G + logistic / ridge | 0.388 | 0.309 | 0.314 | 1.023 | 0.459 | 0.447 | 218.974 | 0.297 | 10.553 | 0.371 | **0.688** | 1.862 |
| DINOv2-L + CatBoost | 0.353 | 0.406 | 0.296 | 1.057 | 0.420 | 0.487 | **209.902** | 0.422 | 10.802 | 0.444 | 0.534 | 1.862 |
| DINOv2-L + MLP | 0.328 | 0.340 | **0.350** | **0.969** | 0.457 | 0.515 | 218.550 | 0.413 | 14.216 | **0.517** | 0.602 | 1.864 |
| DINOv2-reg-L + logistic / ridge | 0.266 | 0.323 | 0.296 | 0.995 | 0.449 | 0.449 | 219.017 | 0.322 | **9.814** | 0.403 | 0.677 | 1.866 |
| DINOv2-L + k-NN k=10 | 0.268 | 0.408 | 0.239 | 1.341 | 0.396 | 0.547 | 219.500 | 0.446 | 10.688 | 0.362 | 0.591 | 1.803 |
| DINOv2-reg-L + k-NN k=10 | **0.397** | **0.452** | 0.271 | 1.635 | 0.402 | **0.560** | 221.511 | 0.375 | 12.886 | 0.422 | 0.601 | 1.788 |
| DINOv2-L + k-NN k=5 | 0.252 | 0.430 | 0.280 | 1.735 | 0.435 | 0.526 | 224.065 | **0.450** | 10.567 | 0.352 | 0.577 | 1.799 |
| DINOv2-reg-L + k-NN k=5 | 0.284 | 0.431 | 0.284 | 2.121 | 0.414 | 0.533 | 221.692 | 0.408 | 12.348 | 0.344 | 0.601 | **1.772** |

## Cross-attribute aggregates

| pipeline | unweighted macro-F1 (cls) | unweighted RMSE (numeric) | unweighted MAE (numeric) |
| --- | --- | --- | --- |
| Baseline (majority / median) | 0.242 | 58.200 | 11.829 |
| DINOv2-G + logistic / ridge | 0.409 | 58.103 | 15.317 |
| DINOv2-L + CatBoost | 0.420 | 55.906 | 15.554 |
| DINOv2-L + MLP | 0.440 | 58.900 | 15.240 |
| DINOv2-reg-L + logistic / ridge | 0.398 | 57.923 | 15.785 |
| DINOv2-L + k-NN k=10 | 0.407 | 58.333 | 13.760 |
| DINOv2-reg-L + k-NN k=10 | 0.435 | 59.455 | 14.364 |
| DINOv2-L + k-NN k=5 | 0.413 | 59.541 | 14.412 |
| DINOv2-reg-L + k-NN k=5 | 0.412 | 59.483 | 13.957 |
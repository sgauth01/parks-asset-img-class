# Baseline (dummy) — Old vs New split

## Classification & boolean attributes — macro-F1 (higher is better)

| Attribute | Old split (80/20) | New split (85/15) | Δ |
| --- | ---: | ---: | ---: |
| Abutment Material | 0.139 | 0.143 | +0.004 |
| Bridge Type | 0.178 | 0.223 | +0.044 |
| Decking Material | 0.193 | 0.240 | +0.046 |
| Has Edge Guard | 0.490 | 0.472 | -0.018 |
| Has Pedestrian Railing | 0.309 | 0.310 | +0.001 |
| Material (Frame, Tank, Body) | 0.085 | 0.093 | +0.008 |
| Structure Material | 0.214 | 0.236 | +0.022 |
| Structure Position | 0.278 | 0.218 | -0.060 |
| **Mean across attributes** | **0.236** | **0.242** | **+0.006** |

## Numeric & count attributes — RMSE (lower is better)

| Attribute | Old split (80/20) | New split (85/15) | Δ |
| --- | ---: | ---: | ---: |
| Fall Height (m) | 4.260 | 1.032 | -3.227 |
| Length (m) | 82.151 | 218.989 | +136.839 |
| Number of Steps | 10.775 | 9.947 | -0.828 |
| Width (m) | 1.411 | 2.831 | +1.420 |

## Numeric & count attributes — MAE (lower is better)

| Attribute | Old split (80/20) | New split (85/15) | Δ |
| --- | ---: | ---: | ---: |
| Fall Height (m) | 1.511 | 0.826 | -0.685 |
| Length (m) | 22.699 | 38.118 | +15.420 |
| Number of Steps | 9.900 | 7.647 | -2.253 |
| Width (m) | 0.556 | 0.724 | +0.169 |

# Baseline on the new split — short answer

**Short version: the baseline barely changes.**

The "baseline" is our dummy classifier — it just picks the most common
training label for each attribute (or the median number, for the numeric
attributes). It doesn't actually *learn* anything from the images, so
which exact rows go into the training set doesn't matter much. The only
thing that moves the numbers is which test rows it gets graded on.

## Cross-attribute summary

Same dummy classifier, scored on two different validation surfaces:

| Split                | Cross-attribute macro-F1 |
|----------------------|:-----:|
| Project 80/20 (seed 42, our shared test set) | **0.236** |
| New suggested split 85/15 (per attribute, seed 48) | **0.242** |
| Δ                   | **+0.006** |

So basically unchanged.

## Per-attribute F1 (classification & boolean — higher is better)

| Attribute                       | Project 80/20 | New suggested 85/15 |   Δ    |
|---------------------------------|:------------:|:----------:|:------:|
| Abutment Material               | 0.139        | 0.143      | +0.004 |
| Bridge Type                     | 0.178        | 0.223      | +0.044 |
| Decking Material                | 0.193        | 0.240      | +0.046 |
| Has Edge Guard                  | 0.490        | 0.472      | −0.018 |
| Has Pedestrian Railing          | 0.309        | 0.310      | +0.001 |
| Material (Frame, Tank, Body)    | 0.085        | 0.093      | +0.008 |
| Structure Material              | 0.214        | 0.236      | +0.022 |
| Structure Position              | 0.278        | 0.218      | −0.060 |
| **Cross-attribute mean**        | **0.236**    | **0.242**  | **+0.006** |

Differences are small and go both ways. There's no real "winner"
between the two splits at the baseline level — it just depends on
which images happen to land in val.

## Per-attribute RMSE (numeric / count — lower is better)

| Attribute        | Project 80/20 | New suggested 85/15 | What's going on |
|------------------|:------------:|:----------:|---|
| Fall Height (m)  | 4.260        | **1.032**  | The new split's val has fewer extreme-fall assets; RMSE *looks* much better but the model is the same. |
| Length (m)       | 82.151       | 218.989    | The new split's val landed the giant ~2,600 m boardwalk. One outlier dominates. |
| Number of Steps  | 10.775       | 9.947      | Tiny val (only 17 rows). Don't read into it. |
| Width (m)        | 1.411        | 2.831      | Same kind of outlier story as length. |

**These swings are NOT a "the new split is better/worse" signal.** The
baseline is identical; the only thing changing is which side of the
asset-grouped split happens to hold the rare extreme assets (2,600 m
boardwalks, 7 m–wide bridges, etc.).

## Recommendation for the numeric attributes

Before partner-facing reporting on `attr_length`, `attr_width`,
`attr_fall_height`, and `attr_number_of_steps`, we should:

1. **Cap outliers** at a sensible quantile (e.g., 99th percentile), OR
2. **Log-transform** the target before computing MAE / RMSE, OR
3. **Restrict to rows where `Length measured by staff == True`** so we
   only evaluate on values that were actually measured (rather than
   field-guesstimated).

Any of those will make the per-split RMSE numbers comparable.

## Reproducibility

To regenerate this file at any time:

```bash
python scripts/compare_baseline_across_splits.py
```

That script reads the per-attribute metric tables already on disk
(`reports/report_tables/per_attribute.csv` for the project split, and
`data/predictions/new_split/per_attribute_metrics.csv` for the new
suggested split) and emits both this file and
`reports/baseline_split_comparison.md` (the fuller technical version
with sample-size annotations).

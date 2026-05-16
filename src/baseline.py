"""Baseline models for tabular park asset attributes.

The goal of these baselines is intentionally modest: estimate what a model
can achieve by learning only the target distribution from the training set.
Stronger image models should beat these numbers, especially when baseline
accuracy is high only because one class dominates the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


DEFAULT_CLASSIFICATION_TARGETS = [
    "attr_abutment_material",
    "attr_bridge_type",
    "attr_decking_material",
    "attr_has_edge_guard",
    "attr_has_pedestrian_railing",
    "attr_material_frame,_tank,_body",
    "attr_structure_material",
    "attr_structure_position",
    "length_bin",
    "width_bin",
    "fall_height_bin",
    "steps_bin",
]


@dataclass(frozen=True)
class ConstantPredictor:
    """Predict the same fitted value for every row.

    This tiny class keeps the baseline behaviour explicit and easy to inspect
    instead of hiding it behind model-specific defaults.
    """

    value: object

    def predict(self, n_rows: int) -> list[object]:
        """Return ``n_rows`` copies of the fitted constant."""
        return [self.value] * n_rows


def first_mode(series: pd.Series) -> object:
    """Return the first modal value after dropping missing labels."""
    modes = series.dropna().mode()
    if modes.empty:
        raise ValueError("Cannot compute a mode for an empty target.")
    return modes.iloc[0]


def existing_columns(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    """Keep requested columns that are present in the input data frame."""
    return [column for column in columns if column in frame.columns]


def evaluate_majority_class(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: Iterable[str] = DEFAULT_CLASSIFICATION_TARGETS,
) -> pd.DataFrame:
    """Evaluate a majority-class classifier for each categorical target.

    Missing labels are excluded per target. Accuracy and weighted/macro F1 are
    logged because accuracy can look strong when the most common class is very
    dominant, while F1 makes that imbalance more visible.
    """
    rows: list[dict[str, object]] = []

    for target in existing_columns(targets, train_df):
        if target not in test_df.columns:
            continue

        y_train = train_df[target].dropna()
        y_test = test_df.loc[test_df[target].notna(), target]
        if y_train.empty or y_test.empty:
            continue

        majority_class = first_mode(y_train)
        predictor = ConstantPredictor(majority_class)
        y_pred = predictor.predict(len(y_test))

        class_counts = y_train.value_counts(dropna=True)
        majority_count = int(class_counts.iloc[0])
        train_label_count = int(class_counts.sum())

        rows.append(
            {
                "attribute": target,
                "task_type": "classification",
                "strategy": "majority_class",
                "prediction": majority_class,
                "n_train_labels": train_label_count,
                "n_test_labels": int(len(y_test)),
                "train_majority_count": majority_count,
                "train_majority_share": majority_count / train_label_count,
                "accuracy": accuracy_score(y_test, y_pred),
                "weighted_f1": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def _make_cv_splitter(y: pd.Series, n_splits: int, random_state: int) -> KFold:
    """Use stratified folds when each class has enough examples."""
    min_class_count = int(y.value_counts().min())
    if min_class_count >= n_splits:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_majority_class(
    df: pd.DataFrame,
    targets: Iterable[str] = DEFAULT_CLASSIFICATION_TARGETS,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate the majority-class baseline on one labelled data set.

    This is the preferred baseline when only ``train.csv`` is available. For
    each target, labels with missing values are excluded, the majority class is
    fit on each training fold, and accuracy/F1 are evaluated on the held-out
    fold. The returned summary contains mean and standard deviation across
    folds; the fold table keeps the per-fold details for debugging.
    """
    fold_rows: list[dict[str, object]] = []

    for target in existing_columns(targets, df):
        labelled = df.loc[df[target].notna(), [target]].reset_index(drop=True)
        if len(labelled) < 2:
            continue

        y = labelled[target]
        target_splits = min(n_splits, len(labelled))
        if target_splits < 2:
            continue

        splitter = _make_cv_splitter(y, target_splits, random_state)
        split_iterator = splitter.split(labelled, y)

        for fold, (train_idx, valid_idx) in enumerate(split_iterator, start=1):
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            majority_class = first_mode(y_train)
            predictor = ConstantPredictor(majority_class)
            y_pred = predictor.predict(len(y_valid))

            class_counts = y_train.value_counts(dropna=True)
            majority_count = int(class_counts.iloc[0])
            train_label_count = int(class_counts.sum())

            fold_rows.append(
                {
                    "attribute": target,
                    "task_type": "classification",
                    "strategy": "majority_class_cv",
                    "fold": fold,
                    "n_folds": target_splits,
                    "prediction": majority_class,
                    "n_train_labels": train_label_count,
                    "n_valid_labels": int(len(y_valid)),
                    "train_majority_count": majority_count,
                    "train_majority_share": majority_count / train_label_count,
                    "accuracy": accuracy_score(y_valid, y_pred),
                    "weighted_f1": f1_score(
                        y_valid, y_pred, average="weighted", zero_division=0
                    ),
                    "macro_f1": f1_score(
                        y_valid, y_pred, average="macro", zero_division=0
                    ),
                }
            )

    fold_results = pd.DataFrame(fold_rows)
    if fold_results.empty:
        return pd.DataFrame(), fold_results

    summary_rows: list[dict[str, object]] = []
    for (attribute, task_type, strategy), group in fold_results.groupby(
        ["attribute", "task_type", "strategy"]
    ):
        summary_rows.append(
            {
                "attribute": attribute,
                "task_type": task_type,
                "strategy": strategy,
                "n_folds": int(group["fold"].max()),
                "n_labels": int(group["n_valid_labels"].sum()),
                "prediction": first_mode(group["prediction"]),
                "train_majority_share_mean": group["train_majority_share"].mean(),
                "train_majority_share_std": group["train_majority_share"].std(ddof=0),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=0),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=0),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=0),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return summary, fold_results


def evaluate_all_baselines(
    df: pd.DataFrame,
    classification_targets: Iterable[str] = DEFAULT_CLASSIFICATION_TARGETS,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return the cross-validated majority-class baseline summary."""
    summary, _ = cross_validate_majority_class(
        df,
        targets=classification_targets,
        n_splits=n_splits,
        random_state=random_state,
    )
    return summary


__all__ = [
    "ConstantPredictor",
    "DEFAULT_CLASSIFICATION_TARGETS",
    "cross_validate_majority_class",
    "evaluate_all_baselines",
    "evaluate_majority_class",
    "first_mode",
]

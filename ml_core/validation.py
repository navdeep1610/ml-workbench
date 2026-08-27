"""Dataset and supervised-learning validation rules."""

from __future__ import annotations

import pandas as pd


MAX_ROWS = 100_000
MAX_COLUMNS = 200


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset is outside the supported MVP scope."""


def validate_dataframe(data: pd.DataFrame) -> None:
    """Validate an uploaded dataframe without modifying user data."""
    if data.empty:
        raise DatasetValidationError("The uploaded CSV does not contain any data rows.")

    if len(data) > MAX_ROWS:
        raise DatasetValidationError(
            f"This version supports at most {MAX_ROWS:,} rows; the file has {len(data):,}."
        )

    if len(data.columns) > MAX_COLUMNS:
        raise DatasetValidationError(
            f"This version supports at most {MAX_COLUMNS} columns; the file has {len(data.columns)}."
        )

    duplicated_names = data.columns[data.columns.duplicated()].tolist()
    if duplicated_names:
        names = ", ".join(map(str, duplicated_names[:5]))
        raise DatasetValidationError(f"Column names must be unique. Repeated names: {names}.")

def validate_problem(data: pd.DataFrame, target: str, task: str) -> None:
    """Validate a selected target and learning task."""
    if target not in data.columns:
        raise DatasetValidationError("Select a valid target column.")

    if len(data.columns) < 2:
        raise DatasetValidationError("The dataset needs at least one feature and one target column.")

    target_values = data[target].dropna()
    if target_values.empty:
        raise DatasetValidationError("The target column does not contain any usable values.")

    unique_targets = target_values.nunique(dropna=True)

    if unique_targets < 2:
        raise DatasetValidationError("The target column must contain at least two distinct values.")

    if task == "Regression" and not pd.api.types.is_numeric_dtype(target_values):
        raise DatasetValidationError("Regression requires a numeric target column.")

    if task == "Classification" and unique_targets > max(50, int(len(data) * 0.2)):
        raise DatasetValidationError(
            "The selected target has too many distinct classes for this classification MVP. "
            "Check whether the task should be regression instead."
        )

    if task == "Classification" and int(target_values.value_counts().min()) < 2:
        raise DatasetValidationError(
            "Every class needs at least two rows so the data can be split safely."
        )

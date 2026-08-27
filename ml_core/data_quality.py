"""Data-quality inspection and safe, explicit normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    """A compact description of common dataset problems."""

    missing_cells: int
    rows_with_missing: int
    duplicate_rows: int
    infinite_values: int
    blank_text_cells: int
    constant_columns: tuple[str, ...]
    high_cardinality_columns: tuple[str, ...]


def inspect_dataframe(data: pd.DataFrame) -> DataQualityReport:
    """Inspect a dataframe without changing user data."""
    text_columns = data.select_dtypes(include=["object", "string"]).columns
    if len(text_columns):
        text_data = data[text_columns].astype("string")
        blank_text_cells = int(text_data.apply(lambda column: column.str.strip().eq("")).sum().sum())
    else:
        blank_text_cells = 0

    numeric = data.select_dtypes(include="number")
    infinite_values = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    constant_columns = tuple(
        str(column) for column in data.columns if data[column].nunique(dropna=False) <= 1
    )

    high_cardinality_columns: list[str] = []
    for column in text_columns:
        unique_count = int(data[column].nunique(dropna=True))
        if unique_count > 50 and unique_count / max(len(data), 1) > 0.2:
            high_cardinality_columns.append(str(column))

    return DataQualityReport(
        missing_cells=int(data.isna().sum().sum()),
        rows_with_missing=int(data.isna().any(axis=1).sum()),
        duplicate_rows=int(data.duplicated().sum()),
        infinite_values=infinite_values,
        blank_text_cells=blank_text_cells,
        constant_columns=constant_columns,
        high_cardinality_columns=tuple(high_cardinality_columns),
    )


def normalize_dataframe(
    data: pd.DataFrame,
    *,
    trim_text: bool = True,
    blank_as_missing: bool = True,
    infinity_as_missing: bool = True,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """Apply reversible-in-spirit cleanup choices to a copy of a dataframe."""
    cleaned = data.copy()
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns

    if trim_text and len(text_columns):
        for column in text_columns:
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    if blank_as_missing and len(text_columns):
        cleaned[text_columns] = cleaned[text_columns].replace(r"^\s*$", pd.NA, regex=True)

    if infinity_as_missing:
        cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


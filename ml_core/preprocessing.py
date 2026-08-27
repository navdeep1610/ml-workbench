"""Preprocessing construction shared by classification and regression models."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(features: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Encode categorical columns and optionally scale numeric columns."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        numeric_transformer: object = StandardScaler() if scale_numeric else "passthrough"
        transformers.append(("numeric", numeric_transformer, numeric_columns))

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


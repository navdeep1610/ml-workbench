"""Preprocessing construction shared by classification and regression models."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    features: pd.DataFrame,
    scale_numeric: bool,
    numeric_missing_strategy: str = "median",
    categorical_missing_strategy: str = "most_frequent",
) -> ColumnTransformer:
    """Impute, encode, and optionally scale features without data leakage."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [
            ("imputer", SimpleImputer(strategy=numeric_missing_strategy))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer: object = Pipeline(numeric_steps)
        transformers.append(("numeric", numeric_transformer, numeric_columns))

    if categorical_columns:
        categorical_imputer = (
            SimpleImputer(strategy="constant", fill_value="Missing")
            if categorical_missing_strategy == "constant"
            else SimpleImputer(strategy="most_frequent")
        )
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", categorical_imputer),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")

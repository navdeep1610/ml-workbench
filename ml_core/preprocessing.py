"""Configurable preprocessing for classification and regression pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler


SCALERS = {
    "None": None,
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
}


def automatic_scaler(model_name: str, models_requiring_scaling: set[str]) -> str:
    """Select a sensible scaler for a model when the user chooses Automatic."""
    return "StandardScaler" if model_name in models_requiring_scaling else "None"


def build_numeric_imputer(
    method: str,
    *,
    simple_strategy: str = "median",
    knn_neighbors: int = 5,
    iterative_max_iter: int = 10,
    add_indicator: bool = False,
    random_state: int = 42,
) -> Any:
    """Create one supported numeric imputer."""
    if method == "SimpleImputer":
        return SimpleImputer(
            strategy=simple_strategy,
            add_indicator=add_indicator,
            keep_empty_features=True,
        )
    if method == "KNNImputer":
        return KNNImputer(
            n_neighbors=knn_neighbors,
            weights="distance",
            add_indicator=add_indicator,
            keep_empty_features=True,
        )
    if method == "IterativeImputer":
        return IterativeImputer(
            max_iter=iterative_max_iter,
            random_state=random_state,
            add_indicator=add_indicator,
            keep_empty_features=False,
            skip_complete=True,
        )
    raise ValueError(f"Unsupported numeric imputer: {method}")


def build_preprocessor(
    features: pd.DataFrame,
    *,
    numeric_imputer: str = "SimpleImputer",
    numeric_simple_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    scaler: str = "None",
    add_indicator: bool = False,
    knn_neighbors: int = 5,
    iterative_max_iter: int = 10,
    random_state: int = 42,
) -> ColumnTransformer:
    """Build a leakage-safe transformer for mixed tabular features."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [
            (
                "imputer",
                build_numeric_imputer(
                    numeric_imputer,
                    simple_strategy=numeric_simple_strategy,
                    knn_neighbors=knn_neighbors,
                    iterative_max_iter=iterative_max_iter,
                    add_indicator=add_indicator,
                    random_state=random_state,
                ),
            )
        ]
        scaler_constructor = SCALERS.get(scaler)
        if scaler not in SCALERS:
            raise ValueError(f"Unsupported scaler: {scaler}")
        if scaler_constructor is not None:
            numeric_steps.append(("scaler", scaler_constructor()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        categorical_imputer = (
            SimpleImputer(
                strategy="constant",
                fill_value="Missing",
                add_indicator=False,
                keep_empty_features=True,
            )
            if categorical_strategy == "constant"
            else SimpleImputer(
                strategy="most_frequent",
                add_indicator=False,
                keep_empty_features=True,
            )
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

    if not transformers:
        raise ValueError("At least one usable feature is required.")

    return ColumnTransformer(transformers=transformers, remainder="drop")

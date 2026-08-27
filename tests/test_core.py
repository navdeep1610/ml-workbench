from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_core.data_quality import inspect_dataframe, normalize_dataframe
from ml_core.models import MODELS_REQUIRING_SCALING, create_estimator, recommended_parameters
from ml_core.preprocessing import build_preprocessor
from ml_core.validation import DatasetValidationError, validate_dataframe, validate_problem


def test_clean_dataframe_is_accepted() -> None:
    data = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "city": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        }
    )
    validate_dataframe(data)
    validate_problem(data, "target", "Classification")


def test_missing_values_are_accepted_and_imputed() -> None:
    data = pd.DataFrame({"feature": [1.0, None], "target": [0, 1]})
    validate_dataframe(data)
    model = Pipeline(
        [
            ("preprocessing", build_preprocessor(data[["feature"]], False)),
            ("model", create_estimator("Classification", "Decision Tree", {})),
        ]
    )
    model.fit(data[["feature"]], data["target"])
    assert len(model.predict(data[["feature"]])) == 2


def test_data_quality_normalization() -> None:
    data = pd.DataFrame(
        {"text": [" A ", "", " A "], "number": [1.0, float("inf"), 1.0]}
    )
    report = inspect_dataframe(data)
    assert report.blank_text_cells == 1
    assert report.infinite_values == 1
    cleaned = normalize_dataframe(data, drop_duplicates=True)
    assert cleaned.loc[0, "text"] == "A"
    assert cleaned.isna().sum().sum() == 2


def test_regression_requires_numeric_target() -> None:
    data = pd.DataFrame({"feature": [1, 2, 3], "target": ["low", "medium", "high"]})
    with pytest.raises(DatasetValidationError, match="numeric target"):
        validate_problem(data, "target", "Regression")


def test_mixed_features_fit_in_pipeline() -> None:
    features = pd.DataFrame({"age": [20, 30, 40, 50], "city": ["A", "B", "A", "B"]})
    target = pd.Series([0, 1, 0, 1])
    estimator = create_estimator(
        "Classification",
        "Logistic Regression",
        {"solver": "lbfgs", "penalty": "l2", "C": 1.0, "max_iter": 1000},
    )
    model = Pipeline(
        [
            ("preprocessing", build_preprocessor(features, "Logistic Regression" in MODELS_REQUIRING_SCALING)),
            ("model", estimator),
        ]
    )
    model.fit(features, target)
    assert model.predict(features).shape == (4,)


def test_recommended_parameters_limit_neighbors_to_training_rows() -> None:
    parameters = recommended_parameters(
        "Classification", "K-Nearest Neighbors", random_state=42, training_rows=3
    )
    assert parameters["n_neighbors"] == 3

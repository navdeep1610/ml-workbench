from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_core.models import MODELS_REQUIRING_SCALING, create_estimator
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


def test_missing_values_are_rejected() -> None:
    data = pd.DataFrame({"feature": [1.0, None], "target": [0, 1]})
    with pytest.raises(DatasetValidationError, match="missing values"):
        validate_dataframe(data)


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

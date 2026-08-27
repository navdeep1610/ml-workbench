"""Supported models and estimator construction."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Gradient Boosting",
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "K-Nearest Neighbors Regressor",
    "Support Vector Regressor",
    "Gradient Boosting Regressor",
]

MODELS_REQUIRING_SCALING = {
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "K-Nearest Neighbors Regressor",
    "Support Vector Regressor",
}


def create_estimator(task: str, model_name: str, parameters: dict[str, Any]) -> BaseEstimator:
    """Create a supported scikit-learn estimator from validated UI parameters."""
    if task == "Classification":
        constructors: dict[str, type[BaseEstimator]] = {
            "Logistic Regression": LogisticRegression,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "K-Nearest Neighbors": KNeighborsClassifier,
            "Support Vector Machine": SVC,
            "Gradient Boosting": GradientBoostingClassifier,
        }
    else:
        constructors = {
            "Linear Regression": LinearRegression,
            "Ridge Regression": Ridge,
            "Lasso Regression": Lasso,
            "Decision Tree Regressor": DecisionTreeRegressor,
            "Random Forest Regressor": RandomForestRegressor,
            "K-Nearest Neighbors Regressor": KNeighborsRegressor,
            "Support Vector Regressor": SVR,
            "Gradient Boosting Regressor": GradientBoostingRegressor,
        }

    try:
        return constructors[model_name](**parameters)
    except KeyError as error:
        raise ValueError(f"Unsupported model: {model_name}") from error


def recommended_parameters(
    task: str, model_name: str, random_state: int, training_rows: int | None = None
) -> dict[str, Any]:
    """Return practical defaults for fair, quick model comparison."""
    neighbors = min(5, max(1, training_rows or 5))
    if task == "Classification":
        defaults: dict[str, dict[str, Any]] = {
            "Logistic Regression": {"max_iter": 1_000, "random_state": random_state},
            "Decision Tree": {"random_state": random_state},
            "Random Forest": {"n_estimators": 200, "random_state": random_state, "n_jobs": -1},
            "K-Nearest Neighbors": {"n_neighbors": neighbors, "n_jobs": -1},
            "Support Vector Machine": {"random_state": random_state},
            "Gradient Boosting": {"random_state": random_state},
        }
    else:
        defaults = {
            "Linear Regression": {"n_jobs": -1},
            "Ridge Regression": {"random_state": random_state},
            "Lasso Regression": {"random_state": random_state, "max_iter": 1_000},
            "Decision Tree Regressor": {"random_state": random_state},
            "Random Forest Regressor": {
                "n_estimators": 200,
                "random_state": random_state,
                "n_jobs": -1,
            },
            "K-Nearest Neighbors Regressor": {"n_neighbors": neighbors, "n_jobs": -1},
            "Support Vector Regressor": {},
            "Gradient Boosting Regressor": {"random_state": random_state},
        }

    try:
        return defaults[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported model: {model_name}") from error

"""Hyperparameter search spaces and scoring rules."""

from __future__ import annotations

from typing import Any


CLASSIFICATION_SCORING = {
    "F1 score (macro)": "f1_macro",
    "F1 score (weighted)": "f1_weighted",
    "Balanced accuracy": "balanced_accuracy",
    "Accuracy": "accuracy",
    "Precision (macro)": "precision_macro",
    "Recall (macro)": "recall_macro",
}

REGRESSION_SCORING = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "R²": "r2",
}


def scoring_options(task: str) -> dict[str, str]:
    """Return user-facing metric names mapped to scikit-learn scorers."""
    return CLASSIFICATION_SCORING if task == "Classification" else REGRESSION_SCORING


def lower_is_better(metric_name: str) -> bool:
    return metric_name in {"RMSE", "MAE", "MSE", "Training time (s)"}


def display_cv_score(metric_name: str, raw_score: float) -> float:
    """Convert negative-loss scorer output to a positive human-readable error."""
    return -raw_score if lower_is_better(metric_name) else raw_score


def parameter_space(
    task: str, model_name: str, training_rows: int | None = None
) -> dict[str, list[Any]]:
    """Return a bounded, cloud-friendly search space for one pipeline."""
    max_neighbors = max(1, min(25, (training_rows or 50) - 1))
    neighbor_values = sorted({value for value in [3, 5, 7, 11, 15] if value <= max_neighbors})
    if not neighbor_values:
        neighbor_values = [1]

    if task == "Classification":
        spaces: dict[str, dict[str, list[Any]]] = {
            "Logistic Regression": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__solver": ["liblinear"],
                "model__penalty": ["l1", "l2"],
                "model__class_weight": [None, "balanced"],
                "model__max_iter": [1_000, 2_000],
            },
            "Decision Tree": {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"],
            },
            "Random Forest": {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 10, 25],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"],
                "model__class_weight": [None, "balanced"],
            },
            "K-Nearest Neighbors": {
                "model__n_neighbors": neighbor_values,
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
            "Support Vector Machine": {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
                "model__class_weight": [None, "balanced"],
            },
            "Gradient Boosting": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.1, 0.2],
                "model__max_depth": [1, 2, 3],
                "model__subsample": [0.7, 0.9, 1.0],
            },
        }
    else:
        spaces = {
            "Linear Regression": {
                "model__fit_intercept": [True, False],
                "model__positive": [True, False],
            },
            "Ridge Regression": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__solver": ["auto", "lsqr", "sag"],
            },
            "Lasso Regression": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "model__selection": ["cyclic", "random"],
                "model__max_iter": [1_000, 5_000],
            },
            "Decision Tree Regressor": {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [None, "sqrt", "log2"],
            },
            "Random Forest Regressor": {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 10, 25],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [1.0, "sqrt", "log2"],
            },
            "K-Nearest Neighbors Regressor": {
                "model__n_neighbors": neighbor_values,
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
            "Support Vector Regressor": {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
                "model__epsilon": [0.01, 0.1, 0.5, 1.0],
            },
            "Gradient Boosting Regressor": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.03, 0.1, 0.2],
                "model__max_depth": [1, 2, 3],
                "model__subsample": [0.7, 0.9, 1.0],
            },
        }

    try:
        return spaces[model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported model: {model_name}") from error


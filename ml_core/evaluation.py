"""Standardized evaluation metrics for trained models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def classification_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    """Return task-level classification metrics using weighted averaging."""
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "Precision (weighted)": float(
            precision_score(actual, predicted, average="weighted", zero_division=0)
        ),
        "Recall (weighted)": float(
            recall_score(actual, predicted, average="weighted", zero_division=0)
        ),
        "F1 score (weighted)": float(
            f1_score(actual, predicted, average="weighted", zero_division=0)
        ),
    }


def regression_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    """Return standard regression metrics."""
    mse = float(mean_squared_error(actual, predicted))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R²": float(r2_score(actual, predicted)),
    }


"""Cloud-ready Streamlit interface for the ML Workbench MVP."""

from __future__ import annotations

import io
import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_core.evaluation import classification_metrics, regression_metrics
from ml_core.models import (
    CLASSIFICATION_MODELS,
    MODELS_REQUIRING_SCALING,
    REGRESSION_MODELS,
    create_estimator,
)
from ml_core.preprocessing import build_preprocessor
from ml_core.validation import DatasetValidationError, validate_dataframe, validate_problem


st.set_page_config(page_title="ML Workbench", page_icon="🧠", layout="wide")


def _class_weight(label: str) -> str | None:
    return None if label == "None" else label


def _optional_depth(label: str, key: str) -> int | None:
    use_limit = st.checkbox("Limit maximum depth", value=False, key=f"{key}_use_depth")
    if not use_limit:
        return None
    return int(
        st.number_input("Maximum depth", min_value=1, max_value=100, value=10, key=f"{key}_depth")
    )


def classification_parameters(model_name: str, random_state: int) -> dict[str, Any]:
    """Render and return valid classification parameters for the selected model."""
    if model_name == "Logistic Regression":
        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        penalties = {
            "lbfgs": ["l2", "None"],
            "liblinear": ["l1", "l2"],
            "saga": ["l1", "l2", "elasticnet", "None"],
        }[solver]
        penalty_label = st.selectbox("Penalty", penalties)
        params: dict[str, Any] = {
            "C": float(st.number_input("C (inverse regularization)", 0.0001, 1000.0, 1.0)),
            "solver": solver,
            "penalty": None if penalty_label == "None" else penalty_label,
            "max_iter": int(st.number_input("Maximum iterations", 100, 20_000, 1_000, step=100)),
            "class_weight": _class_weight(st.selectbox("Class weight", ["None", "balanced"])),
            "random_state": random_state,
        }
        if penalty_label == "elasticnet":
            params["l1_ratio"] = float(st.slider("L1 ratio", 0.0, 1.0, 0.5, 0.05))
        return params

    if model_name == "Decision Tree":
        return {
            "criterion": st.selectbox("Split criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": _optional_depth(model_name, "dtc"),
            "min_samples_split": int(st.number_input("Minimum samples to split", 2, 100, 2)),
            "min_samples_leaf": int(st.number_input("Minimum samples per leaf", 1, 100, 1)),
            "max_features": st.selectbox("Maximum features", [None, "sqrt", "log2"]),
            "class_weight": _class_weight(st.selectbox("Class weight", ["None", "balanced"])),
            "random_state": random_state,
        }

    if model_name == "Random Forest":
        return {
            "n_estimators": int(st.number_input("Number of trees", 10, 1_000, 200, step=10)),
            "criterion": st.selectbox("Split criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": _optional_depth(model_name, "rfc"),
            "min_samples_split": int(st.number_input("Minimum samples to split", 2, 100, 2)),
            "min_samples_leaf": int(st.number_input("Minimum samples per leaf", 1, 100, 1)),
            "max_features": st.selectbox("Maximum features", ["sqrt", "log2", None]),
            "bootstrap": st.checkbox("Bootstrap samples", value=True),
            "class_weight": _class_weight(st.selectbox("Class weight", ["None", "balanced"])),
            "random_state": random_state,
            "n_jobs": -1,
        }

    if model_name == "K-Nearest Neighbors":
        metric = st.selectbox("Distance metric", ["minkowski", "euclidean", "manhattan"])
        params = {
            "n_neighbors": int(st.number_input("Number of neighbors", 1, 100, 5)),
            "weights": st.selectbox("Neighbor weights", ["uniform", "distance"]),
            "metric": metric,
            "n_jobs": -1,
        }
        if metric == "minkowski":
            params["p"] = int(st.selectbox("Minkowski power", [1, 2, 3]))
        return params

    if model_name == "Support Vector Machine":
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        params = {
            "C": float(st.number_input("C (regularization)", 0.0001, 1000.0, 1.0)),
            "kernel": kernel,
            "gamma": st.selectbox("Gamma", ["scale", "auto"]),
            "class_weight": _class_weight(st.selectbox("Class weight", ["None", "balanced"])),
            "probability": st.checkbox("Enable probability estimates", value=False),
            "random_state": random_state,
        }
        if kernel == "poly":
            params["degree"] = int(st.number_input("Polynomial degree", 1, 10, 3))
        return params

    return {
        "n_estimators": int(st.number_input("Number of boosting stages", 10, 1_000, 100, step=10)),
        "learning_rate": float(st.number_input("Learning rate", 0.001, 1.0, 0.1)),
        "max_depth": int(st.number_input("Maximum tree depth", 1, 20, 3)),
        "subsample": float(st.slider("Training sample fraction", 0.1, 1.0, 1.0, 0.05)),
        "max_features": st.selectbox("Maximum features", [None, "sqrt", "log2"]),
        "random_state": random_state,
    }


def regression_parameters(model_name: str, random_state: int) -> dict[str, Any]:
    """Render and return regression parameters for the selected model."""
    if model_name == "Linear Regression":
        return {
            "fit_intercept": st.checkbox("Fit intercept", value=True),
            "n_jobs": -1,
        }

    if model_name == "Ridge Regression":
        return {
            "alpha": float(st.number_input("Alpha (regularization)", 0.0001, 10_000.0, 1.0)),
            "solver": st.selectbox("Solver", ["auto", "lsqr", "sag", "saga"]),
            "max_iter": int(st.number_input("Maximum iterations", 100, 20_000, 1_000, step=100)),
            "tol": float(st.number_input("Tolerance", 0.000001, 0.1, 0.0001, format="%.6f")),
            "random_state": random_state,
        }

    if model_name == "Lasso Regression":
        return {
            "alpha": float(st.number_input("Alpha (regularization)", 0.0001, 10_000.0, 1.0)),
            "max_iter": int(st.number_input("Maximum iterations", 100, 20_000, 1_000, step=100)),
            "tol": float(st.number_input("Tolerance", 0.000001, 0.1, 0.0001, format="%.6f")),
            "selection": st.selectbox("Coefficient selection", ["cyclic", "random"]),
            "random_state": random_state,
        }

    if model_name == "Decision Tree Regressor":
        return {
            "criterion": st.selectbox(
                "Split criterion", ["squared_error", "friedman_mse", "absolute_error"]
            ),
            "max_depth": _optional_depth(model_name, "dtr"),
            "min_samples_split": int(st.number_input("Minimum samples to split", 2, 100, 2)),
            "min_samples_leaf": int(st.number_input("Minimum samples per leaf", 1, 100, 1)),
            "max_features": st.selectbox("Maximum features", [None, "sqrt", "log2"]),
            "random_state": random_state,
        }

    if model_name == "Random Forest Regressor":
        return {
            "n_estimators": int(st.number_input("Number of trees", 10, 1_000, 200, step=10)),
            "criterion": st.selectbox(
                "Split criterion", ["squared_error", "friedman_mse", "absolute_error"]
            ),
            "max_depth": _optional_depth(model_name, "rfr"),
            "min_samples_split": int(st.number_input("Minimum samples to split", 2, 100, 2)),
            "min_samples_leaf": int(st.number_input("Minimum samples per leaf", 1, 100, 1)),
            "max_features": st.selectbox("Maximum features", [1.0, "sqrt", "log2"]),
            "bootstrap": st.checkbox("Bootstrap samples", value=True),
            "random_state": random_state,
            "n_jobs": -1,
        }

    if model_name == "K-Nearest Neighbors Regressor":
        metric = st.selectbox("Distance metric", ["minkowski", "euclidean", "manhattan"])
        params = {
            "n_neighbors": int(st.number_input("Number of neighbors", 1, 100, 5)),
            "weights": st.selectbox("Neighbor weights", ["uniform", "distance"]),
            "metric": metric,
            "n_jobs": -1,
        }
        if metric == "minkowski":
            params["p"] = int(st.selectbox("Minkowski power", [1, 2, 3]))
        return params

    if model_name == "Support Vector Regressor":
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        params = {
            "C": float(st.number_input("C (regularization)", 0.0001, 1000.0, 1.0)),
            "kernel": kernel,
            "gamma": st.selectbox("Gamma", ["scale", "auto"]),
            "epsilon": float(st.number_input("Epsilon", 0.0, 100.0, 0.1)),
        }
        if kernel == "poly":
            params["degree"] = int(st.number_input("Polynomial degree", 1, 10, 3))
        return params

    return {
        "loss": st.selectbox("Loss", ["squared_error", "absolute_error", "huber"]),
        "n_estimators": int(st.number_input("Number of boosting stages", 10, 1_000, 100, step=10)),
        "learning_rate": float(st.number_input("Learning rate", 0.001, 1.0, 0.1)),
        "max_depth": int(st.number_input("Maximum tree depth", 1, 20, 3)),
        "subsample": float(st.slider("Training sample fraction", 0.1, 1.0, 1.0, 0.05)),
        "max_features": st.selectbox("Maximum features", [None, "sqrt", "log2"]),
        "random_state": random_state,
    }


def metric_cards(metrics: dict[str, float], elapsed: float) -> None:
    columns = st.columns(min(len(metrics) + 1, 5))
    for column, (name, value) in zip(columns, metrics.items()):
        column.metric(name, f"{value:.4f}")
    columns[-1].metric("Training time", f"{elapsed:.2f} s")


def classification_charts(actual: pd.Series, predicted: np.ndarray, model: Pipeline) -> None:
    labels = list(model.named_steps["model"].classes_)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    figure = px.imshow(
        matrix,
        x=[str(label) for label in labels],
        y=[str(label) for label in labels],
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Confusion matrix",
    )
    st.plotly_chart(figure, use_container_width=True)

    if len(labels) != 2:
        return

    estimator = model.named_steps["model"]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(st.session_state["x_test"])[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = model.decision_function(st.session_state["x_test"])
    else:
        return

    false_positive_rate, true_positive_rate, _ = roc_curve(actual, scores, pos_label=labels[1])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_figure = go.Figure()
    roc_figure.add_trace(
        go.Scatter(x=false_positive_rate, y=true_positive_rate, name=f"Model (AUC={roc_auc:.3f})")
    )
    roc_figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"dash": "dash"})
    )
    roc_figure.update_layout(
        title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate"
    )
    st.plotly_chart(roc_figure, use_container_width=True)


def regression_charts(actual: pd.Series, predicted: np.ndarray) -> None:
    chart_data = pd.DataFrame({"Actual": actual.to_numpy(), "Predicted": predicted})
    left, right = st.columns(2)
    actual_figure = px.scatter(chart_data, x="Actual", y="Predicted", title="Actual vs predicted")
    minimum = float(min(chart_data.min()))
    maximum = float(max(chart_data.max()))
    actual_figure.add_trace(
        go.Scatter(x=[minimum, maximum], y=[minimum, maximum], mode="lines", name="Ideal")
    )
    left.plotly_chart(actual_figure, use_container_width=True)

    chart_data["Residual"] = chart_data["Actual"] - chart_data["Predicted"]
    residual_figure = px.scatter(
        chart_data, x="Predicted", y="Residual", title="Residuals vs predicted"
    )
    residual_figure.add_hline(y=0, line_dash="dash")
    right.plotly_chart(residual_figure, use_container_width=True)


def main() -> None:
    st.title("🧠 ML Workbench")
    st.write(
        "Upload a clean CSV, configure a classification or regression model, "
        "and evaluate it without writing code."
    )
    st.info(
        "MVP rule: datasets containing missing values are rejected. "
        "The cloud limit is 25 MB, 100,000 rows, and 200 columns."
    )

    uploaded_file = st.file_uploader("Upload a clean CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.stop()

    try:
        data = pd.read_csv(uploaded_file)
        validate_dataframe(data)
    except (DatasetValidationError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(str(error))
        st.stop()

    st.subheader("1. Review the dataset")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Rows", f"{len(data):,}")
    summary_columns[1].metric("Columns", len(data.columns))
    summary_columns[2].metric("Numeric columns", len(data.select_dtypes(include="number").columns))
    summary_columns[3].metric("Duplicate rows", int(data.duplicated().sum()))
    st.dataframe(data.head(100), use_container_width=True)

    st.subheader("2. Define the problem")
    target = st.selectbox("Target column", data.columns.tolist())
    task = st.radio("Problem type", ["Classification", "Regression"], horizontal=True)

    try:
        validate_problem(data, target, task)
    except DatasetValidationError as error:
        st.warning(str(error))
        st.stop()

    model_options = CLASSIFICATION_MODELS if task == "Classification" else REGRESSION_MODELS
    model_name = st.selectbox("Model", model_options)

    settings_left, settings_right = st.columns(2)
    with settings_left:
        test_percentage = st.slider("Test-data percentage", 10, 40, 20, 5)
    with settings_right:
        random_state = int(st.number_input("Random seed", 0, 1_000_000, 42))

    st.subheader("3. Configure hyperparameters")
    st.caption(
        "These settings are specific to the selected model. Numeric scaling is applied automatically "
        "when the algorithm is sensitive to feature scale."
    )
    with st.container(border=True):
        parameters = (
            classification_parameters(model_name, random_state)
            if task == "Classification"
            else regression_parameters(model_name, random_state)
        )

    if not st.button("Train and evaluate model", type="primary", use_container_width=True):
        st.stop()

    features = data.drop(columns=[target])
    target_values = data[target]
    estimator = create_estimator(task, model_name, parameters)
    preprocessor = build_preprocessor(features, model_name in MODELS_REQUIRING_SCALING)
    model = Pipeline([("preprocessing", preprocessor), ("model", estimator)])

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target_values,
            test_size=test_percentage / 100,
            random_state=random_state,
            stratify=target_values if task == "Classification" else None,
        )
        st.session_state["x_test"] = x_test
        start_time = time.perf_counter()
        with st.spinner(f"Training {model_name}..."):
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
        elapsed = time.perf_counter() - start_time
    except (ValueError, TypeError, MemoryError) as error:
        st.error(f"Training could not be completed: {error}")
        st.stop()

    st.subheader("4. Model results")
    if task == "Classification":
        metrics = classification_metrics(y_test, predicted)
        metric_cards(metrics, elapsed)
        classification_charts(y_test, predicted, model)
    else:
        metrics = regression_metrics(y_test, predicted)
        metric_cards(metrics, elapsed)
        regression_charts(y_test, predicted)

    predictions = x_test.copy()
    predictions.insert(0, "source_row", predictions.index)
    predictions["actual_target"] = y_test
    predictions["predicted_target"] = predicted

    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    report = {
        "task": task,
        "model": model_name,
        "target": target,
        "test_percentage": test_percentage,
        "random_state": random_state,
        "hyperparameters": parameters,
        "metrics": metrics,
        "training_seconds": elapsed,
    }

    download_left, download_middle, download_right = st.columns(3)
    download_left.download_button(
        "Download predictions",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_middle.download_button(
        "Download model report",
        json.dumps(report, indent=2, default=str),
        file_name="model_report.json",
        mime="application/json",
        use_container_width=True,
    )
    download_right.download_button(
        "Download trained model",
        model_buffer.getvalue(),
        file_name="trained_model.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()


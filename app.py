"""Cloud-ready Streamlit interface for the ML Workbench MVP."""

from __future__ import annotations

import io
import importlib
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

from ml_core import data_quality as data_quality_module
from ml_core import evaluation as evaluation_module
from ml_core import models as models_module
from ml_core import preprocessing as preprocessing_module
from ml_core import validation as validation_module


# Streamlit can rerun app.py in the same process after a cloud update. Reloading local
# modules ensures a deployment never mixes a new interface with cached old helpers.
for local_module in (
    data_quality_module,
    evaluation_module,
    models_module,
    preprocessing_module,
    validation_module,
):
    importlib.reload(local_module)

inspect_dataframe = data_quality_module.inspect_dataframe
normalize_dataframe = data_quality_module.normalize_dataframe
classification_metrics = evaluation_module.classification_metrics
regression_metrics = evaluation_module.regression_metrics
CLASSIFICATION_MODELS = models_module.CLASSIFICATION_MODELS
MODELS_REQUIRING_SCALING = models_module.MODELS_REQUIRING_SCALING
REGRESSION_MODELS = models_module.REGRESSION_MODELS
create_estimator = models_module.create_estimator
recommended_parameters = models_module.recommended_parameters
build_preprocessor = preprocessing_module.build_preprocessor
DatasetValidationError = validation_module.DatasetValidationError
validate_dataframe = validation_module.validate_dataframe
validate_problem = validation_module.validate_problem


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
    values = [*metrics.items(), ("Training time", elapsed)]
    for start in range(0, len(values), 4):
        row = values[start : start + 4]
        columns = st.columns(len(row))
        for column, (name, value) in zip(columns, row):
            suffix = " s" if name == "Training time" else ""
            column.metric(name, f"{value:.4f}{suffix}")


def build_model_pipeline(
    features: pd.DataFrame,
    task: str,
    model_name: str,
    parameters: dict[str, Any],
    numeric_missing_strategy: str,
    categorical_missing_strategy: str,
) -> Pipeline:
    """Create one end-to-end model with leakage-safe preprocessing."""
    estimator = create_estimator(task, model_name, parameters)
    preprocessor = build_preprocessor(
        features,
        model_name in MODELS_REQUIRING_SCALING,
        numeric_missing_strategy,
        categorical_missing_strategy,
    )
    return Pipeline([("preprocessing", preprocessor), ("model", estimator)])


def render_model_comparison(
    features: pd.DataFrame,
    target_values: pd.Series,
    task: str,
    model_names: list[str],
    test_percentage: int,
    random_state: int,
    numeric_missing_strategy: str,
    categorical_missing_strategy: str,
    primary_metric: str,
) -> None:
    """Train selected models on one split and display a fair leaderboard."""
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target_values,
            test_size=test_percentage / 100,
            random_state=random_state,
            stratify=target_values if task == "Classification" else None,
        )
    except ValueError as error:
        st.error(f"The data could not be split safely: {error}")
        return

    result_rows: list[dict[str, Any]] = []
    trained_models: dict[str, Pipeline] = {}
    failures: list[str] = []
    progress = st.progress(0, text="Preparing model comparison...")

    for index, model_name in enumerate(model_names):
        progress.progress(index / len(model_names), text=f"Training {model_name}...")
        try:
            parameters = recommended_parameters(
                task, model_name, random_state, training_rows=len(x_train)
            )
            model = build_model_pipeline(
                features,
                task,
                model_name,
                parameters,
                numeric_missing_strategy,
                categorical_missing_strategy,
            )
            start_time = time.perf_counter()
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            elapsed = time.perf_counter() - start_time
            metrics = (
                classification_metrics(y_test, predicted)
                if task == "Classification"
                else regression_metrics(y_test, predicted)
            )
            result_rows.append({"Model": model_name, **metrics, "Training time (s)": elapsed})
            trained_models[model_name] = model
        except (ValueError, TypeError, MemoryError) as error:
            failures.append(f"{model_name}: {error}")

    progress.progress(1.0, text="Comparison complete")
    if not result_rows:
        st.error("None of the selected models could be trained.")
        for failure in failures:
            st.caption(failure)
        return

    results = pd.DataFrame(result_rows)
    lower_is_better = primary_metric in {"MAE", "MSE", "RMSE", "Training time (s)"}
    results = results.sort_values(primary_metric, ascending=lower_is_better).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))

    st.subheader("5. Model comparison")
    best_model_name = str(results.iloc[0]["Model"])
    st.success(f"Best by {primary_metric}: {best_model_name}")
    st.dataframe(
        results.style.format(
            {column: "{:.4f}" for column in results.select_dtypes(include="number").columns if column != "Rank"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart = px.bar(
        results,
        x="Model",
        y=primary_metric,
        color="Model",
        title=f"Model comparison by {primary_metric}",
        text_auto=".4f",
    )
    chart.update_layout(showlegend=False)
    st.plotly_chart(chart, use_container_width=True)
    st.caption(
        "Every model used the same train/test rows. Comparison uses recommended baseline "
        "hyperparameters; tune the winner in Single model mode before making a final decision."
    )

    if failures:
        with st.expander(f"{len(failures)} model(s) could not be compared"):
            for failure in failures:
                st.write(f"- {failure}")

    comparison_csv = results.to_csv(index=False).encode("utf-8")
    best_model_buffer = io.BytesIO()
    joblib.dump(trained_models[best_model_name], best_model_buffer)
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download comparison table",
        comparison_csv,
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_right.download_button(
        f"Download best model ({best_model_name})",
        best_model_buffer.getvalue(),
        file_name="best_model.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )


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
        "Upload a CSV, review data-quality problems, and train or compare classification "
        "and regression models without writing code."
    )
    st.info(
        "Missing feature values can now be handled automatically. Rows with a missing target "
        "are excluded because the correct answer cannot be safely guessed. The cloud limit is "
        "25 MB, 100,000 rows, and 200 columns."
    )

    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.stop()

    try:
        raw_data = pd.read_csv(uploaded_file)
        validate_dataframe(raw_data)
    except (DatasetValidationError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(str(error))
        st.stop()

    st.subheader("1. Inspect and clean the dataset")
    raw_report = inspect_dataframe(raw_data)
    summary_columns = st.columns(5)
    summary_columns[0].metric("Rows", f"{len(raw_data):,}")
    summary_columns[1].metric("Missing cells", f"{raw_report.missing_cells:,}")
    summary_columns[2].metric("Blank text cells", f"{raw_report.blank_text_cells:,}")
    summary_columns[3].metric("Infinite values", f"{raw_report.infinite_values:,}")
    summary_columns[4].metric("Duplicate rows", f"{raw_report.duplicate_rows:,}")

    with st.expander("Data-cleaning options", expanded=any([
        raw_report.missing_cells,
        raw_report.blank_text_cells,
        raw_report.infinite_values,
        raw_report.duplicate_rows,
    ])):
        st.caption("The original uploaded file is never changed. These choices affect this run only.")
        cleaning_left, cleaning_right = st.columns(2)
        with cleaning_left:
            trim_text = st.checkbox(
                "Remove spaces before and after text", value=True,
                help="For example, 'Delhi ' and 'Delhi' become the same category.",
            )
            blank_as_missing = st.checkbox(
                "Treat blank text as missing", value=True,
                help="Blank strings will use the missing-value strategy selected below.",
            )
        with cleaning_right:
            infinity_as_missing = st.checkbox(
                "Treat +∞ and -∞ as missing", value=True,
                help="Most machine-learning algorithms cannot train with infinite numeric values.",
            )
            drop_duplicates = st.checkbox(
                "Remove duplicate rows", value=raw_report.duplicate_rows > 0,
                help="Only completely identical rows are removed.",
            )

    data = normalize_dataframe(
        raw_data,
        trim_text=trim_text,
        blank_as_missing=blank_as_missing,
        infinity_as_missing=infinity_as_missing,
        drop_duplicates=drop_duplicates,
    )
    cleaned_report = inspect_dataframe(data)
    removed_rows = len(raw_data) - len(data)
    if removed_rows:
        st.success(f"Removed {removed_rows:,} duplicate row(s) for this run.")

    if cleaned_report.constant_columns:
        st.warning(
            "Constant columns do not help prediction and should normally be excluded: "
            + ", ".join(cleaned_report.constant_columns)
        )
    if cleaned_report.high_cardinality_columns:
        st.warning(
            "These text columns contain many different values and may be IDs or free text: "
            + ", ".join(cleaned_report.high_cardinality_columns)
            + ". Consider excluding them to avoid a very large model."
        )

    st.dataframe(data.head(100), use_container_width=True)

    st.subheader("2. Define the problem")
    target = st.selectbox("Target column", data.columns.tolist())
    task = st.radio("Problem type", ["Classification", "Regression"], horizontal=True)

    missing_target_rows = int(data[target].isna().sum())
    if missing_target_rows:
        st.warning(
            f"{missing_target_rows:,} row(s) have no target value and will be excluded from training. "
            "Feature values can be imputed, but target values should not be guessed."
        )
    model_data = data.dropna(subset=[target]).copy()

    try:
        validate_problem(model_data, target, task)
    except DatasetValidationError as error:
        st.warning(str(error))
        st.stop()

    model_options = CLASSIFICATION_MODELS if task == "Classification" else REGRESSION_MODELS

    suggested_exclusions = [
        column
        for column in [*cleaned_report.constant_columns, *cleaned_report.high_cardinality_columns]
        if column != target
    ]
    excluded_columns = st.multiselect(
        "Columns to exclude from training",
        [column for column in model_data.columns if column != target],
        default=suggested_exclusions,
        help="Exclude IDs, names, free text, leakage columns, or fields that will not exist when predicting.",
    )
    features = model_data.drop(columns=[target, *excluded_columns])
    target_values = model_data[target]
    if features.shape[1] == 0:
        st.error("Keep at least one feature column for training.")
        st.stop()

    feature_missing_count = int(features.isna().sum().sum())
    numeric_missing_strategy = "median"
    categorical_missing_strategy = "most_frequent"
    with st.expander(
        "Missing-value strategy",
        expanded=feature_missing_count > 0,
    ):
        if feature_missing_count:
            st.write(f"The selected feature columns contain **{feature_missing_count:,} missing cells**.")
        else:
            st.success("The selected feature columns do not contain missing values.")
        missing_left, missing_right = st.columns(2)
        with missing_left:
            numeric_strategy_label = st.selectbox(
                "Numeric columns",
                ["Median (recommended)", "Mean", "Most frequent"],
                help="Median is usually safest because it is less affected by extreme values.",
            )
            numeric_missing_strategy = {
                "Median (recommended)": "median",
                "Mean": "mean",
                "Most frequent": "most_frequent",
            }[numeric_strategy_label]
        with missing_right:
            categorical_strategy_label = st.selectbox(
                "Text/category columns",
                ["Most frequent", "Create a 'Missing' category"],
            )
            categorical_missing_strategy = (
                "constant"
                if categorical_strategy_label == "Create a 'Missing' category"
                else "most_frequent"
            )
        st.caption(
            "The imputer learns values from training rows only, then applies them to test rows. "
            "This prevents data leakage."
        )

    st.subheader("3. Choose how to train")
    workflow = st.radio(
        "Training mode",
        ["Single model", "Compare models"],
        horizontal=True,
        help="Compare mode uses the same split and preprocessing rules for every selected model.",
    )

    settings_left, settings_right = st.columns(2)
    with settings_left:
        test_percentage = st.slider("Test-data percentage", 10, 40, 20, 5)
    with settings_right:
        random_state = int(st.number_input("Random seed", 0, 1_000_000, 42))

    if workflow == "Compare models":
        default_models = (
            ["Logistic Regression", "Decision Tree", "Random Forest"]
            if task == "Classification"
            else ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
        )
        selected_models = st.multiselect(
            "Models to compare",
            model_options,
            default=default_models,
            help="Start with three models. SVM and gradient boosting can take longer on large datasets.",
        )
        metric_options = (
            [
                "F1 score (weighted)",
                "Balanced accuracy",
                "Accuracy",
                "F1 score (macro)",
                "Precision (weighted)",
                "Recall (weighted)",
            ]
            if task == "Classification"
            else ["RMSE", "MAE", "R²", "MSE"]
        )
        primary_metric = st.selectbox(
            "Metric used to rank models",
            metric_options,
            help=(
                "F1 balances precision and recall. Balanced accuracy or macro F1 is often better "
                "when one class is much more common than another."
                if task == "Classification"
                else "RMSE penalizes large mistakes more strongly; MAE is easier to interpret; higher R² is better."
            ),
        )
        if not selected_models:
            st.warning("Select at least one model to compare.")
            st.stop()
        if not st.button("Compare selected models", type="primary", use_container_width=True):
            st.stop()
        render_model_comparison(
            features,
            target_values,
            task,
            selected_models,
            test_percentage,
            random_state,
            numeric_missing_strategy,
            categorical_missing_strategy,
            primary_metric,
        )
        st.stop()

    model_name = st.selectbox("Model", model_options)

    st.subheader("4. Configure hyperparameters")
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

    model = build_model_pipeline(
        features,
        task,
        model_name,
        parameters,
        numeric_missing_strategy,
        categorical_missing_strategy,
    )

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

    st.subheader("5. Model results")
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
        "excluded_columns": excluded_columns,
        "numeric_missing_strategy": numeric_missing_strategy,
        "categorical_missing_strategy": categorical_missing_strategy,
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

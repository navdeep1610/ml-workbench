"""White-glass, cloud-ready interface for the ML Workbench."""

from __future__ import annotations

import hashlib
import importlib
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
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from ml_core import data_quality as data_quality_module
from ml_core import evaluation as evaluation_module
from ml_core import models as models_module
from ml_core import preprocessing as preprocessing_module
from ml_core import tuning as tuning_module
from ml_core import validation as validation_module


for local_module in (
    data_quality_module,
    evaluation_module,
    models_module,
    preprocessing_module,
    tuning_module,
    validation_module,
):
    importlib.reload(local_module)

inspect_dataframe = data_quality_module.inspect_dataframe
normalize_dataframe = data_quality_module.normalize_dataframe
classification_metrics = evaluation_module.classification_metrics
regression_metrics = evaluation_module.regression_metrics
CLASSIFICATION_MODELS = models_module.CLASSIFICATION_MODELS
REGRESSION_MODELS = models_module.REGRESSION_MODELS
MODELS_REQUIRING_SCALING = models_module.MODELS_REQUIRING_SCALING
create_estimator = models_module.create_estimator
recommended_parameters = models_module.recommended_parameters
automatic_scaler = preprocessing_module.automatic_scaler
build_preprocessor = preprocessing_module.build_preprocessor
display_cv_score = tuning_module.display_cv_score
lower_is_better = tuning_module.lower_is_better
parameter_space = tuning_module.parameter_space
scoring_options = tuning_module.scoring_options
DatasetValidationError = validation_module.DatasetValidationError
validate_dataframe = validation_module.validate_dataframe
validate_problem = validation_module.validate_problem


st.set_page_config(page_title="ML Workbench", page_icon="✨", layout="wide")


WORKFLOW_STEPS = ["1 · Prepare", "2 · Model", "3 · Visualize"]


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --workbench-ink: #15223a;
            --workbench-muted: #68758d;
            --workbench-blue: #5268f5;
            --workbench-violet: #8d69e7;
            --workbench-teal: #129b88;
            --workbench-border: #dfe6f1;
        }
        .stApp {
            color: var(--workbench-ink);
            background:
                radial-gradient(circle at 7% 5%, #e8f8ff 0, #e8f8ff 5%, transparent 22%),
                radial-gradient(circle at 94% 8%, #eee9ff 0, #eee9ff 6%, transparent 24%),
                linear-gradient(145deg, #fcfeff 0%, #f3f7ff 55%, #faf6ff 100%);
        }
        [data-testid="stHeader"] { background: rgba(255,255,255,.78); }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 4rem;
        }
        .workbench-hero {
            background: rgba(255,255,255,.94);
            border: 1px solid var(--workbench-border);
            border-radius: 22px;
            box-shadow: 0 20px 55px rgba(62,79,120,.12);
            margin-bottom: 1rem;
            padding: 1.2rem 1.35rem;
        }
        .workbench-brand { align-items: center; display: flex; gap: .7rem; }
        .workbench-logo {
            align-items: center;
            background: linear-gradient(135deg, var(--workbench-blue), var(--workbench-violet));
            border-radius: .75rem;
            box-shadow: 0 8px 20px rgba(82,104,245,.24);
            color: white;
            display: inline-flex;
            font-size: 1.1rem;
            height: 2.4rem;
            justify-content: center;
            width: 2.4rem;
        }
        .workbench-kicker {
            color: var(--workbench-blue);
            font-size: .72rem;
            font-weight: 600;
            letter-spacing: .11em;
            margin: 0 0 .3rem;
            text-transform: uppercase;
        }
        .workbench-hero h1 { font-size: clamp(1.6rem, 4vw, 2.35rem); letter-spacing: -.04em; margin: .8rem 0 .25rem; }
        .workbench-hero p { color: var(--workbench-muted); margin: 0; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,.96);
            border-color: var(--workbench-border) !important;
            border-radius: 18px;
            box-shadow: inset 0 1px 0 white, 0 12px 34px rgba(73,89,128,.09);
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #ffffff, #f7f9ff);
            border: 1px solid var(--workbench-border);
            border-radius: 14px;
            padding: .8rem 1rem;
        }
        div[data-testid="stMetricLabel"] { color: var(--workbench-muted); }
        .stButton > button, .stDownloadButton > button {
            border-color: var(--workbench-border);
            border-radius: 11px;
            min-height: 2.6rem;
        }
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--workbench-blue), #735ee9);
            border: 0;
            box-shadow: 0 9px 20px rgba(82,104,245,.2);
        }
        div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stTextInput input {
            background: rgba(255,255,255,.98);
            border-color: var(--workbench-border);
            border-radius: 10px;
        }
        div[data-testid="stDataFrame"] { border: 1px solid var(--workbench-border); border-radius: 13px; overflow: hidden; }
        div[data-testid="stExpander"] { background: rgba(255,255,255,.82); border-color: var(--workbench-border); border-radius: 13px; }
        div[data-testid="stAlert"] { border-radius: 13px; }
        div[data-testid="stSegmentedControl"] { background: rgba(255,255,255,.94); border: 1px solid var(--workbench-border); border-radius: 15px; padding: .3rem; }
        div[data-testid="stSegmentedControl"] button { min-height: 2.75rem; }
        hr { border-color: var(--workbench-border); }
        .workbench-section-copy { color: var(--workbench-muted); margin-top: -.55rem; }
        .workbench-pill {
            background: #edf1ff;
            border-radius: 999px;
            color: var(--workbench-blue);
            display: inline-block;
            font-size: .75rem;
            padding: .35rem .65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="workbench-hero">
          <div class="workbench-brand"><span class="workbench-logo">✦</span><strong>ML Workbench</strong></div>
          <h1>A complete experiment in three clear steps</h1>
          <p>Prepare the data, fit one model or compare several, then understand the result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_pending_step(step: str) -> None:
    st.session_state["pending_workflow_step"] = step


def initialize_state() -> None:
    if "workflow_step" not in st.session_state:
        st.session_state["workflow_step"] = WORKFLOW_STEPS[0]
    pending = st.session_state.pop("pending_workflow_step", None)
    if pending:
        st.session_state["workflow_step"] = pending


def clear_experiment() -> None:
    st.session_state.pop("experiment_result", None)


def class_weight(label: str) -> str | None:
    return None if label == "None" else label


def optional_depth(prefix: str, default: int = 10) -> int | None:
    if not st.checkbox("Limit maximum depth", value=False, key=f"{prefix}_use_depth"):
        return None
    return int(
        st.number_input(
            "Maximum depth", min_value=1, max_value=100, value=default, key=f"{prefix}_depth"
        )
    )


def classification_parameters(model_name: str, random_state: int, prefix: str) -> dict[str, Any]:
    if model_name == "Logistic Regression":
        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"], key=f"{prefix}_solver")
        penalties = {
            "lbfgs": ["l2", "None"],
            "liblinear": ["l1", "l2"],
            "saga": ["l1", "l2", "elasticnet", "None"],
        }[solver]
        penalty_label = st.selectbox("Penalty", penalties, key=f"{prefix}_penalty")
        parameters: dict[str, Any] = {
            "C": float(
                st.number_input(
                    "C (inverse regularization)",
                    min_value=0.0001,
                    max_value=1_000.0,
                    value=1.0,
                    key=f"{prefix}_c",
                )
            ),
            "solver": solver,
            "penalty": None if penalty_label == "None" else penalty_label,
            "max_iter": int(
                st.number_input(
                    "Maximum iterations",
                    min_value=100,
                    max_value=20_000,
                    value=1_000,
                    step=100,
                    key=f"{prefix}_max_iter",
                )
            ),
            "class_weight": class_weight(
                st.selectbox("Class weight", ["None", "balanced"], key=f"{prefix}_weight")
            ),
            "random_state": random_state,
        }
        if penalty_label == "elasticnet":
            parameters["l1_ratio"] = float(
                st.slider("L1 ratio", 0.0, 1.0, 0.5, 0.05, key=f"{prefix}_l1")
            )
        return parameters

    if model_name == "Decision Tree":
        return {
            "criterion": st.selectbox(
                "Split criterion", ["gini", "entropy", "log_loss"], key=f"{prefix}_criterion"
            ),
            "max_depth": optional_depth(prefix),
            "min_samples_split": int(
                st.number_input(
                    "Minimum samples to split", 2, 100, 2, key=f"{prefix}_min_split"
                )
            ),
            "min_samples_leaf": int(
                st.number_input("Minimum samples per leaf", 1, 100, 1, key=f"{prefix}_min_leaf")
            ),
            "max_features": st.selectbox(
                "Maximum features", [None, "sqrt", "log2"], key=f"{prefix}_features"
            ),
            "class_weight": class_weight(
                st.selectbox("Class weight", ["None", "balanced"], key=f"{prefix}_weight")
            ),
            "random_state": random_state,
        }

    if model_name == "Random Forest":
        return {
            "n_estimators": int(
                st.number_input(
                    "Number of trees", 10, 1_000, 200, step=10, key=f"{prefix}_trees"
                )
            ),
            "criterion": st.selectbox(
                "Split criterion", ["gini", "entropy", "log_loss"], key=f"{prefix}_criterion"
            ),
            "max_depth": optional_depth(prefix),
            "min_samples_split": int(
                st.number_input(
                    "Minimum samples to split", 2, 100, 2, key=f"{prefix}_min_split"
                )
            ),
            "min_samples_leaf": int(
                st.number_input("Minimum samples per leaf", 1, 100, 1, key=f"{prefix}_min_leaf")
            ),
            "max_features": st.selectbox(
                "Maximum features", ["sqrt", "log2", None], key=f"{prefix}_features"
            ),
            "bootstrap": st.checkbox("Bootstrap samples", value=True, key=f"{prefix}_bootstrap"),
            "class_weight": class_weight(
                st.selectbox("Class weight", ["None", "balanced"], key=f"{prefix}_weight")
            ),
            "random_state": random_state,
            "n_jobs": -1,
        }

    if model_name == "K-Nearest Neighbors":
        metric = st.selectbox(
            "Distance metric", ["minkowski", "euclidean", "manhattan"], key=f"{prefix}_metric"
        )
        parameters = {
            "n_neighbors": int(
                st.number_input("Number of neighbors", 1, 100, 5, key=f"{prefix}_neighbors")
            ),
            "weights": st.selectbox(
                "Neighbor weights", ["uniform", "distance"], key=f"{prefix}_weights"
            ),
            "metric": metric,
            "n_jobs": -1,
        }
        if metric == "minkowski":
            parameters["p"] = int(
                st.selectbox("Minkowski power", [1, 2, 3], key=f"{prefix}_power")
            )
        return parameters

    if model_name == "Support Vector Machine":
        kernel = st.selectbox(
            "Kernel", ["rbf", "linear", "poly", "sigmoid"], key=f"{prefix}_kernel"
        )
        parameters = {
            "C": float(
                st.number_input(
                    "C (regularization)", 0.0001, 1_000.0, 1.0, key=f"{prefix}_c"
                )
            ),
            "kernel": kernel,
            "gamma": st.selectbox("Gamma", ["scale", "auto"], key=f"{prefix}_gamma"),
            "class_weight": class_weight(
                st.selectbox("Class weight", ["None", "balanced"], key=f"{prefix}_weight")
            ),
            "random_state": random_state,
        }
        if kernel == "poly":
            parameters["degree"] = int(
                st.number_input("Polynomial degree", 1, 10, 3, key=f"{prefix}_degree")
            )
        return parameters

    return {
        "n_estimators": int(
            st.number_input(
                "Number of boosting stages", 10, 1_000, 100, step=10, key=f"{prefix}_stages"
            )
        ),
        "learning_rate": float(
            st.number_input("Learning rate", 0.001, 1.0, 0.1, key=f"{prefix}_rate")
        ),
        "max_depth": int(st.number_input("Maximum tree depth", 1, 20, 3, key=f"{prefix}_depth")),
        "subsample": float(
            st.slider("Training sample fraction", 0.1, 1.0, 1.0, 0.05, key=f"{prefix}_sample")
        ),
        "max_features": st.selectbox(
            "Maximum features", [None, "sqrt", "log2"], key=f"{prefix}_features"
        ),
        "random_state": random_state,
    }


def regression_parameters(model_name: str, random_state: int, prefix: str) -> dict[str, Any]:
    if model_name == "Linear Regression":
        return {
            "fit_intercept": st.checkbox("Fit intercept", value=True, key=f"{prefix}_intercept"),
            "positive": st.checkbox("Positive coefficients only", value=False, key=f"{prefix}_positive"),
            "n_jobs": -1,
        }
    if model_name in {"Ridge Regression", "Lasso Regression"}:
        parameters: dict[str, Any] = {
            "alpha": float(
                st.number_input(
                    "Alpha (regularization)", 0.0001, 10_000.0, 1.0, key=f"{prefix}_alpha"
                )
            ),
            "max_iter": int(
                st.number_input(
                    "Maximum iterations", 100, 20_000, 1_000, step=100, key=f"{prefix}_max_iter"
                )
            ),
            "tol": float(
                st.number_input(
                    "Tolerance",
                    0.000001,
                    0.1,
                    0.0001,
                    format="%.6f",
                    key=f"{prefix}_tol",
                )
            ),
            "random_state": random_state,
        }
        if model_name == "Ridge Regression":
            parameters["solver"] = st.selectbox(
                "Solver", ["auto", "lsqr", "sag", "saga"], key=f"{prefix}_solver"
            )
        else:
            parameters["selection"] = st.selectbox(
                "Coefficient selection", ["cyclic", "random"], key=f"{prefix}_selection"
            )
        return parameters

    if model_name in {"Decision Tree Regressor", "Random Forest Regressor"}:
        parameters = {
            "criterion": st.selectbox(
                "Split criterion",
                ["squared_error", "friedman_mse", "absolute_error"],
                key=f"{prefix}_criterion",
            ),
            "max_depth": optional_depth(prefix),
            "min_samples_split": int(
                st.number_input(
                    "Minimum samples to split", 2, 100, 2, key=f"{prefix}_min_split"
                )
            ),
            "min_samples_leaf": int(
                st.number_input("Minimum samples per leaf", 1, 100, 1, key=f"{prefix}_min_leaf")
            ),
            "max_features": st.selectbox(
                "Maximum features",
                [1.0, "sqrt", "log2"] if model_name == "Random Forest Regressor" else [None, "sqrt", "log2"],
                key=f"{prefix}_features",
            ),
            "random_state": random_state,
        }
        if model_name == "Random Forest Regressor":
            parameters.update(
                {
                    "n_estimators": int(
                        st.number_input(
                            "Number of trees", 10, 1_000, 200, step=10, key=f"{prefix}_trees"
                        )
                    ),
                    "bootstrap": st.checkbox(
                        "Bootstrap samples", value=True, key=f"{prefix}_bootstrap"
                    ),
                    "n_jobs": -1,
                }
            )
        return parameters

    if model_name == "K-Nearest Neighbors Regressor":
        metric = st.selectbox(
            "Distance metric", ["minkowski", "euclidean", "manhattan"], key=f"{prefix}_metric"
        )
        parameters = {
            "n_neighbors": int(
                st.number_input("Number of neighbors", 1, 100, 5, key=f"{prefix}_neighbors")
            ),
            "weights": st.selectbox(
                "Neighbor weights", ["uniform", "distance"], key=f"{prefix}_weights"
            ),
            "metric": metric,
            "n_jobs": -1,
        }
        if metric == "minkowski":
            parameters["p"] = int(
                st.selectbox("Minkowski power", [1, 2, 3], key=f"{prefix}_power")
            )
        return parameters

    if model_name == "Support Vector Regressor":
        kernel = st.selectbox(
            "Kernel", ["rbf", "linear", "poly", "sigmoid"], key=f"{prefix}_kernel"
        )
        parameters = {
            "C": float(st.number_input("C", 0.0001, 1_000.0, 1.0, key=f"{prefix}_c")),
            "kernel": kernel,
            "gamma": st.selectbox("Gamma", ["scale", "auto"], key=f"{prefix}_gamma"),
            "epsilon": float(
                st.number_input("Epsilon", 0.0, 100.0, 0.1, key=f"{prefix}_epsilon")
            ),
        }
        if kernel == "poly":
            parameters["degree"] = int(
                st.number_input("Polynomial degree", 1, 10, 3, key=f"{prefix}_degree")
            )
        return parameters

    return {
        "loss": st.selectbox(
            "Loss", ["squared_error", "absolute_error", "huber"], key=f"{prefix}_loss"
        ),
        "n_estimators": int(
            st.number_input(
                "Number of boosting stages", 10, 1_000, 100, step=10, key=f"{prefix}_stages"
            )
        ),
        "learning_rate": float(
            st.number_input("Learning rate", 0.001, 1.0, 0.1, key=f"{prefix}_rate")
        ),
        "max_depth": int(st.number_input("Maximum tree depth", 1, 20, 3, key=f"{prefix}_depth")),
        "subsample": float(
            st.slider("Training sample fraction", 0.1, 1.0, 1.0, 0.05, key=f"{prefix}_sample")
        ),
        "max_features": st.selectbox(
            "Maximum features", [None, "sqrt", "log2"], key=f"{prefix}_features"
        ),
        "random_state": random_state,
    }


def render_manual_parameters(
    task: str, model_name: str, random_state: int, prefix: str
) -> dict[str, Any]:
    return (
        classification_parameters(model_name, random_state, prefix)
        if task == "Classification"
        else regression_parameters(model_name, random_state, prefix)
    )


def read_stored_dataframe() -> pd.DataFrame | None:
    file_bytes = st.session_state.get("dataset_bytes")
    if not file_bytes:
        return None
    return pd.read_csv(io.BytesIO(file_bytes))


def normalized_from_config(raw_data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return normalize_dataframe(
        raw_data,
        trim_text=config["trim_text"],
        blank_as_missing=config["blank_as_missing"],
        infinity_as_missing=config["infinity_as_missing"],
        drop_duplicates=config["drop_duplicates"],
        numeric_columns=config["numeric_like_columns"],
    )


def modeling_data(
    raw_data: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    cleaned = normalized_from_config(raw_data, config)
    cleaned = cleaned.dropna(subset=[config["target"]]).copy()
    features = cleaned.drop(columns=[config["target"], *config["excluded_columns"]])
    target_values = cleaned[config["target"]]
    return features, target_values, cleaned


def feature_recipe(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = config["target"]
    excluded = set(config["excluded_columns"])
    for column in data.columns:
        series = data[column]
        missing = int(series.isna().sum())
        if column == target:
            role = "Target"
            treatment = "Remove rows with missing target"
            transformation = "Not transformed"
        elif column in excluded:
            role = "Excluded"
            treatment = "Trim / normalize only"
            transformation = "Not used for training"
        elif pd.api.types.is_numeric_dtype(series):
            role = "Numeric"
            if config["numeric_imputer"] == "SimpleImputer":
                treatment = f"SimpleImputer · {config['numeric_simple_strategy']}"
            elif config["numeric_imputer"] == "KNNImputer":
                treatment = f"KNNImputer · {config['knn_neighbors']} neighbors"
            else:
                treatment = f"IterativeImputer · {config['iterative_max_iter']} rounds"
            transformation = config["scaler"]
        else:
            role = "Category"
            treatment = (
                "Dedicated Missing category"
                if config["categorical_strategy"] == "constant"
                else "Most frequent"
            )
            transformation = "OneHotEncoder"
        rows.append(
            {
                "Feature": str(column),
                "Role": role,
                "Detected type": str(series.dtype),
                "Missing": missing,
                "Imputation / cleaning": treatment,
                "Transformation": transformation,
            }
        )
    return pd.DataFrame(rows)


def save_preparation(config: dict[str, Any]) -> None:
    st.session_state["prepared_config"] = config
    clear_experiment()
    set_pending_step(WORKFLOW_STEPS[1])


def render_prepare() -> None:
    st.header("Prepare the dataset")
    st.markdown(
        '<p class="workbench-section-copy">Clean deterministic problems, configure learned preprocessing, and inspect every feature.</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload or replace a CSV dataset", type=["csv"], key="dataset_upload"
    )
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
        fingerprint = hashlib.sha256(uploaded_bytes).hexdigest()
        if fingerprint != st.session_state.get("dataset_fingerprint"):
            st.session_state["dataset_bytes"] = uploaded_bytes
            st.session_state["dataset_fingerprint"] = fingerprint
            st.session_state["dataset_name"] = uploaded.name
            st.session_state.pop("prepared_config", None)
            clear_experiment()

    try:
        raw_data = read_stored_dataframe()
        if raw_data is None:
            st.info("Upload a CSV to begin the preparation recipe.")
            return
        validate_dataframe(raw_data)
    except (DatasetValidationError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        st.error(str(error))
        return

    report = inspect_dataframe(raw_data)
    with st.container(border=True):
        st.subheader("Dataset overview")
        st.caption(
            f"{st.session_state.get('dataset_name', 'dataset.csv')} · {len(raw_data):,} rows · "
            f"{len(raw_data.columns)} columns"
        )
        columns = st.columns(4)
        columns[0].metric("Missing cells", f"{report.missing_cells:,}")
        columns[1].metric("Blank text", f"{report.blank_text_cells:,}")
        columns[2].metric("Infinite values", f"{report.infinite_values:,}")
        columns[3].metric("Duplicate rows", f"{report.duplicate_rows:,}")
        st.dataframe(raw_data.head(100), width="stretch", hide_index=True)

    existing = st.session_state.get("prepared_config", {})
    with st.container(border=True):
        st.subheader("1. Define roles and deterministic cleaning")
        left, right = st.columns(2)
        with left:
            target = st.selectbox(
                "Target column",
                raw_data.columns.tolist(),
                index=raw_data.columns.tolist().index(existing.get("target", raw_data.columns[-1]))
                if existing.get("target", raw_data.columns[-1]) in raw_data.columns
                else len(raw_data.columns) - 1,
            )
            task = st.segmented_control(
                "Problem type",
                ["Classification", "Regression"],
                default=existing.get("task", "Classification"),
                key="prep_task",
            )
        with right:
            numeric_like_columns = st.multiselect(
                "Convert numeric-like text columns",
                raw_data.columns.tolist(),
                default=[
                    column
                    for column in existing.get(
                        "numeric_like_columns", list(report.numeric_like_columns)
                    )
                    if column in raw_data.columns
                ],
                help="Values that cannot be converted become missing and are handled by the imputer.",
            )
            suggested_exclusions = [
                column
                for column in [*report.constant_columns, *report.high_cardinality_columns]
                if column != target
            ]
            excluded_columns = st.multiselect(
                "Exclude columns from training",
                [column for column in raw_data.columns if column != target],
                default=[
                    column
                    for column in existing.get("excluded_columns", suggested_exclusions)
                    if column in raw_data.columns and column != target
                ],
                help="Exclude identifiers, names, free text, leakage fields, and columns unavailable at prediction time.",
            )

        clean_a, clean_b, clean_c, clean_d = st.columns(4)
        trim_text = clean_a.checkbox(
            "Trim text spaces", value=existing.get("trim_text", True), key="prep_trim"
        )
        blank_as_missing = clean_b.checkbox(
            "Blanks are missing", value=existing.get("blank_as_missing", True), key="prep_blank"
        )
        infinity_as_missing = clean_c.checkbox(
            "Infinity is missing", value=existing.get("infinity_as_missing", True), key="prep_infinity"
        )
        drop_duplicates = clean_d.checkbox(
            "Remove duplicates",
            value=existing.get("drop_duplicates", report.duplicate_rows > 0),
            key="prep_duplicates",
        )

    interim_config = {
        "target": target,
        "task": task or "Classification",
        "numeric_like_columns": numeric_like_columns,
        "excluded_columns": excluded_columns,
        "trim_text": trim_text,
        "blank_as_missing": blank_as_missing,
        "infinity_as_missing": infinity_as_missing,
        "drop_duplicates": drop_duplicates,
    }
    normalized = normalized_from_config(raw_data, interim_config)

    with st.container(border=True):
        st.subheader("2. Configure the preparation recipe")
        numeric_left, numeric_right = st.columns(2)
        with numeric_left:
            numeric_imputer = st.selectbox(
                "Numeric imputer",
                ["SimpleImputer", "KNNImputer", "IterativeImputer"],
                index=["SimpleImputer", "KNNImputer", "IterativeImputer"].index(
                    existing.get("numeric_imputer", "SimpleImputer")
                ),
            )
            numeric_simple_strategy = existing.get("numeric_simple_strategy", "median")
            knn_neighbors = int(existing.get("knn_neighbors", 5))
            iterative_max_iter = int(existing.get("iterative_max_iter", 10))
            if numeric_imputer == "SimpleImputer":
                numeric_simple_strategy = st.selectbox(
                    "SimpleImputer strategy",
                    ["median", "mean", "most_frequent", "constant"],
                    index=["median", "mean", "most_frequent", "constant"].index(
                        numeric_simple_strategy
                    ),
                )
            elif numeric_imputer == "KNNImputer":
                knn_neighbors = int(
                    st.slider("KNN neighbors", 2, 25, min(max(knn_neighbors, 2), 25))
                )
                st.caption("KNN imputation is slower and uses relationships between rows.")
            else:
                iterative_max_iter = int(
                    st.slider(
                        "Iterative imputation rounds", 5, 30, min(max(iterative_max_iter, 5), 30), 5
                    )
                )
                st.caption("IterativeImputer is experimental and can be expensive on wide datasets.")
        with numeric_right:
            categorical_strategy_label = st.selectbox(
                "Categorical imputer",
                ["Most frequent", "Create a Missing category"],
                index=1 if existing.get("categorical_strategy") == "constant" else 0,
            )
            categorical_strategy = (
                "constant" if categorical_strategy_label == "Create a Missing category" else "most_frequent"
            )
            scaler = st.selectbox(
                "Numeric scaling",
                ["Automatic", "None", "StandardScaler", "RobustScaler", "MinMaxScaler"],
                index=["Automatic", "None", "StandardScaler", "RobustScaler", "MinMaxScaler"].index(
                    existing.get("scaler", "Automatic")
                ),
            )
            add_indicator = st.checkbox(
                "Add missing-value indicators",
                value=existing.get("add_indicator", True),
                help="Adds binary features that record whether a value was originally missing.",
            )

        split_left, split_right = st.columns(2)
        with split_left:
            test_percentage = int(
                st.slider(
                    "Protected test-data percentage",
                    10,
                    40,
                    int(existing.get("test_percentage", 20)),
                    5,
                )
            )
        with split_right:
            random_state = int(
                st.number_input(
                    "Random seed", 0, 1_000_000, int(existing.get("random_state", 42))
                )
            )

    config = {
        **interim_config,
        "numeric_imputer": numeric_imputer,
        "numeric_simple_strategy": numeric_simple_strategy,
        "knn_neighbors": knn_neighbors,
        "iterative_max_iter": iterative_max_iter,
        "categorical_strategy": categorical_strategy,
        "scaler": scaler,
        "add_indicator": add_indicator,
        "test_percentage": test_percentage,
        "random_state": random_state,
    }

    with st.container(border=True):
        st.subheader("3. Review every feature")
        st.caption("This table is the full preparation recipe that will travel with the trained model.")
        st.dataframe(feature_recipe(normalized, config), width="stretch", hide_index=True)

    valid = True
    try:
        model_ready = normalized.dropna(subset=[target])
        validate_problem(model_ready, target, config["task"])
        usable_features = model_ready.drop(columns=[target, *excluded_columns])
        if usable_features.shape[1] == 0:
            raise DatasetValidationError("Keep at least one input feature for training.")
    except DatasetValidationError as error:
        valid = False
        st.error(str(error))

    st.button(
        "Save preparation recipe and continue",
        type="primary",
        width="stretch",
        disabled=not valid,
        on_click=save_preparation,
        args=(config,),
    )


def build_model_pipeline(
    features: pd.DataFrame,
    task: str,
    model_name: str,
    parameters: dict[str, Any],
    config: dict[str, Any],
) -> Pipeline:
    selected_scaler = config["scaler"]
    if selected_scaler == "Automatic":
        selected_scaler = automatic_scaler(model_name, MODELS_REQUIRING_SCALING)
    preprocessor = build_preprocessor(
        features,
        numeric_imputer=config["numeric_imputer"],
        numeric_simple_strategy=config["numeric_simple_strategy"],
        categorical_strategy=config["categorical_strategy"],
        scaler=selected_scaler,
        add_indicator=config["add_indicator"],
        knn_neighbors=config["knn_neighbors"],
        iterative_max_iter=config["iterative_max_iter"],
        random_state=config["random_state"],
    )
    return Pipeline(
        [("preprocessing", preprocessor), ("model", create_estimator(task, model_name, parameters))]
    )


def primary_metric_value(metrics: dict[str, float], metric_name: str) -> float:
    try:
        return float(metrics[metric_name])
    except KeyError as error:
        raise ValueError(f"Metric {metric_name} is not available for this task.") from error


def run_experiment(
    features: pd.DataFrame,
    target_values: pd.Series,
    config: dict[str, Any],
    experiment_mode: str,
    tuning_mode: str,
    model_names: list[str],
    manual_parameters: dict[str, dict[str, Any]],
    primary_metric: str,
    cv_folds: int,
    search_method: str,
    search_budget: int,
) -> dict[str, Any]:
    task = config["task"]
    random_state = config["random_state"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target_values,
        test_size=config["test_percentage"] / 100,
        random_state=random_state,
        stratify=target_values if task == "Classification" else None,
    )
    needs_cv = tuning_mode == "Automatic search" or experiment_mode == "Compare models"
    cv: StratifiedKFold | KFold | None = None
    if needs_cv and task == "Classification":
        smallest_training_class = int(y_train.value_counts().min())
        effective_folds = min(cv_folds, smallest_training_class)
        if effective_folds < 2:
            raise ValueError(
                "The smallest class needs at least two training rows for cross-validation."
            )
        cv = StratifiedKFold(
            n_splits=effective_folds, shuffle=True, random_state=random_state
        )
    elif needs_cv:
        effective_folds = min(cv_folds, len(x_train))
        if effective_folds < 2:
            raise ValueError("At least two training rows are required for cross-validation.")
        cv = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    scorer = scoring_options(task)[primary_metric]
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    metric_sets: dict[str, dict[str, float]] = {}
    best_parameters: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    progress = st.progress(0, text="Preparing experiment...")

    for index, model_name in enumerate(model_names):
        progress.progress(index / len(model_names), text=f"Training {model_name}...")
        try:
            parameters = (
                manual_parameters[model_name]
                if tuning_mode == "Manual hyperparameters"
                else recommended_parameters(task, model_name, random_state, len(x_train))
            )
            pipeline = build_model_pipeline(features, task, model_name, parameters, config)
            start = time.perf_counter()
            cv_score: float | None = None
            cv_std: float | None = None
            chosen_parameters = parameters

            if tuning_mode == "Automatic search":
                search_space = parameter_space(task, model_name, len(x_train))
                if search_method == "GridSearchCV":
                    search_space = {
                        name: values[:2] for name, values in search_space.items()
                    }
                    search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
                        pipeline,
                        search_space,
                        scoring=scorer,
                        cv=cv,
                        n_jobs=1,
                        refit=True,
                        error_score=np.nan,
                    )
                else:
                    search = RandomizedSearchCV(
                        pipeline,
                        search_space,
                        n_iter=search_budget,
                        scoring=scorer,
                        cv=cv,
                        random_state=random_state,
                        n_jobs=1,
                        refit=True,
                        error_score=np.nan,
                    )
                search.fit(x_train, y_train)
                model = search.best_estimator_
                cv_score = display_cv_score(primary_metric, float(search.best_score_))
                best_index = int(search.best_index_)
                cv_std = float(search.cv_results_["std_test_score"][best_index])
                chosen_parameters = {
                    key.removeprefix("model__"): value
                    for key, value in search.best_params_.items()
                }
            else:
                if experiment_mode == "Compare models":
                    cv_scores = cross_val_score(
                        pipeline, x_train, y_train, scoring=scorer, cv=cv, n_jobs=1
                    )
                    cv_score = display_cv_score(primary_metric, float(np.mean(cv_scores)))
                    cv_std = float(np.std(cv_scores))
                pipeline.fit(x_train, y_train)
                model = pipeline

            predicted = model.predict(x_test)
            elapsed = time.perf_counter() - start
            metrics = (
                classification_metrics(y_test, predicted)
                if task == "Classification"
                else regression_metrics(y_test, predicted)
            )
            row: dict[str, Any] = {
                "Model": model_name,
                **metrics,
                "CV score": cv_score,
                "CV std": cv_std,
                "Training time (s)": elapsed,
            }
            rows.append(row)
            fitted[model_name] = model
            predictions[model_name] = predicted
            metric_sets[model_name] = metrics
            best_parameters[model_name] = chosen_parameters
        except (ValueError, TypeError, MemoryError) as error:
            failures.append(f"{model_name}: {error}")

    progress.progress(1.0, text="Experiment complete")
    if not rows:
        raise ValueError("None of the selected models could be trained. " + " | ".join(failures))

    comparison = pd.DataFrame(rows)
    comparison["Test primary score"] = comparison.apply(
        lambda row: primary_metric_value(row.to_dict(), primary_metric), axis=1
    )
    comparison["Ranking score"] = comparison["CV score"].fillna(
        comparison["Test primary score"]
    )
    comparison = comparison.sort_values(
        "Ranking score", ascending=lower_is_better(primary_metric)
    ).reset_index(drop=True)
    comparison.insert(0, "Rank", np.arange(1, len(comparison) + 1))
    best_model_name = str(comparison.iloc[0]["Model"])
    best_model = fitted[best_model_name]
    best_prediction = predictions[best_model_name]
    best_metrics = metric_sets[best_model_name]
    predictions_frame = x_test.copy()
    predictions_frame.insert(0, "source_row", predictions_frame.index)
    predictions_frame["actual_target"] = y_test
    predictions_frame["predicted_target"] = best_prediction

    return {
        "task": task,
        "target": config["target"],
        "experiment_mode": experiment_mode,
        "tuning_mode": tuning_mode,
        "primary_metric": primary_metric,
        "model_name": best_model_name,
        "model": best_model,
        "metrics": best_metrics,
        "comparison": comparison,
        "best_parameters": best_parameters[best_model_name],
        "all_best_parameters": best_parameters,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "predicted": best_prediction,
        "predictions_frame": predictions_frame,
        "failures": failures,
        "config": config,
    }


def render_model(raw_data: pd.DataFrame, config: dict[str, Any]) -> None:
    features, target_values, _ = modeling_data(raw_data, config)
    task = config["task"]
    options = CLASSIFICATION_MODELS if task == "Classification" else REGRESSION_MODELS
    st.header("Fit the model")
    st.markdown(
        '<p class="workbench-section-copy">Choose a single result or a fair comparison, with manual control or cross-validated search.</p>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.subheader("Experiment design")
        mode_left, mode_right = st.columns(2)
        with mode_left:
            experiment_mode = st.segmented_control(
                "Result type",
                ["Single model", "Compare models"],
                default="Single model",
                key="experiment_mode",
            )
        with mode_right:
            tuning_mode = st.segmented_control(
                "Hyperparameter control",
                ["Manual hyperparameters", "Automatic search"],
                default="Manual hyperparameters",
                key="tuning_mode",
            )

        metric_map = scoring_options(task)
        settings_a, settings_b = st.columns(2)
        with settings_a:
            primary_metric = st.selectbox(
                "Primary metric",
                list(metric_map),
                index=0,
                help="This metric ranks models. All other task metrics are still reported.",
            )
        with settings_b:
            cv_folds = int(st.selectbox("Cross-validation folds", [3, 5, 10], index=1))

    manual_parameters: dict[str, dict[str, Any]] = {}
    search_method = "RandomizedSearchCV"
    search_budget = 25
    if experiment_mode == "Single model":
        model_names = [st.selectbox("Model", options)]
    else:
        defaults = (
            ["Logistic Regression", "Random Forest", "Gradient Boosting"]
            if task == "Classification"
            else ["Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor"]
        )
        model_names = st.multiselect("Models to compare", options, default=defaults)

    if not model_names:
        st.warning("Select at least one model.")
        return

    if tuning_mode == "Automatic search" and len(model_names) > 4:
        st.warning(
            "Automatic cloud search supports at most four models per experiment. "
            "Run another comparison for the remaining algorithms."
        )
        return

    if tuning_mode == "Manual hyperparameters":
        with st.container(border=True):
            st.subheader("Manual hyperparameters")
            st.caption("Open each model and choose its settings. Nothing is searched automatically.")
            for model_name in model_names:
                safe_prefix = "manual_" + "_".join(model_name.lower().split())
                with st.expander(model_name, expanded=len(model_names) == 1):
                    manual_parameters[model_name] = render_manual_parameters(
                        task, model_name, config["random_state"], safe_prefix
                    )
    else:
        with st.container(border=True):
            st.subheader("Automatic search")
            search_left, search_right = st.columns(2)
            with search_left:
                search_method = st.selectbox(
                    "Search strategy", ["RandomizedSearchCV", "GridSearchCV"]
                )
            with search_right:
                budget_label = st.selectbox(
                    "Random-search budget",
                    ["Quick · 10 candidates", "Balanced · 25 candidates", "Thorough · 50 candidates"],
                    index=1,
                    disabled=search_method == "GridSearchCV",
                )
                search_budget = {
                    "Quick · 10 candidates": 10,
                    "Balanced · 25 candidates": 25,
                    "Thorough · 50 candidates": 50,
                }[budget_label]
            estimated_fits = len(model_names) * cv_folds * (
                search_budget if search_method == "RandomizedSearchCV" else 50
            )
            st.info(
                f"Estimated workload: about {estimated_fits:,} model fits. GridSearchCV may run "
                "more combinations depending on the selected algorithms."
            )
            for model_name in model_names:
                with st.expander(f"{model_name} search space"):
                    space = parameter_space(task, model_name, len(features))
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Hyperparameter": [key.removeprefix("model__") for key in space],
                                "Candidate values": [", ".join(map(str, value)) for value in space.values()],
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

    if st.button(
        "Run experiment",
        type="primary",
        width="stretch",
    ):
        try:
            result = run_experiment(
                features,
                target_values,
                config,
                experiment_mode or "Single model",
                tuning_mode or "Manual hyperparameters",
                model_names,
                manual_parameters,
                primary_metric,
                cv_folds,
                search_method,
                search_budget,
            )
            st.session_state["experiment_result"] = result
            set_pending_step(WORKFLOW_STEPS[2])
            st.rerun()
        except (ValueError, TypeError, MemoryError) as error:
            st.error(f"The experiment could not be completed: {error}")


def metric_cards(metrics: dict[str, float]) -> None:
    items = list(metrics.items())
    for start in range(0, len(items), 3):
        row = items[start : start + 3]
        columns = st.columns(len(row))
        for column, (name, value) in zip(columns, row):
            column.metric(name, f"{value:.4f}")


def probability_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray | None:
    estimator = model.named_steps["model"]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
    if hasattr(estimator, "decision_function"):
        values = model.decision_function(features)
        if np.ndim(values) == 1:
            return np.asarray(values)
    return None


def render_classification_results(result: dict[str, Any]) -> None:
    actual = result["y_test"]
    predicted = result["predicted"]
    model = result["model"]
    labels = list(model.named_steps["model"].classes_)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    left, right = st.columns([0.9, 1.1])
    with left:
        matrix_figure = px.imshow(
            matrix,
            x=[str(label) for label in labels],
            y=[str(label) for label in labels],
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            title="Confusion matrix",
            color_continuous_scale=["#f4f6ff", "#5268f5"],
        )
        st.plotly_chart(matrix_figure, width="stretch")
    with right:
        report = classification_report(
            actual, predicted, output_dict=True, zero_division=0
        )
        report_frame = pd.DataFrame(report).T.reset_index(names="Class / average")
        st.markdown("#### Per-class report")
        st.dataframe(
            report_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "precision": st.column_config.NumberColumn(format="%.4f"),
                "recall": st.column_config.NumberColumn(format="%.4f"),
                "f1-score": st.column_config.NumberColumn(format="%.4f"),
                "support": st.column_config.NumberColumn(format="%.0f"),
            },
        )

    if len(labels) != 2:
        return
    scores = probability_scores(model, result["x_test"])
    if scores is None:
        return
    positive_label = labels[1]
    false_positive_rate, true_positive_rate, _ = roc_curve(
        actual, scores, pos_label=positive_label
    )
    precision, recall, _ = precision_recall_curve(actual, scores, pos_label=positive_label)
    curve_left, curve_right = st.columns(2)
    with curve_left:
        roc_auc = auc(false_positive_rate, true_positive_rate)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=false_positive_rate,
                y=true_positive_rate,
                name=f"Model · AUC {roc_auc:.3f}",
                line={"color": "#5268f5"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line={"dash": "dash", "color": "#9aa5b8"},
            )
        )
        figure.update_layout(
            title="ROC curve",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
        )
        st.plotly_chart(figure, width="stretch")
    with curve_right:
        pr_auc = auc(recall, precision)
        figure = go.Figure(
            go.Scatter(
                x=recall,
                y=precision,
                name=f"Model · AUC {pr_auc:.3f}",
                line={"color": "#129b88"},
            )
        )
        figure.update_layout(
            title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision"
        )
        st.plotly_chart(figure, width="stretch")


def render_regression_results(result: dict[str, Any]) -> None:
    chart_data = pd.DataFrame(
        {"Actual": result["y_test"].to_numpy(), "Predicted": result["predicted"]}
    )
    left, right = st.columns(2)
    with left:
        figure = px.scatter(
            chart_data,
            x="Actual",
            y="Predicted",
            title="Actual versus predicted",
            color_discrete_sequence=["#5268f5"],
        )
        minimum = float(min(chart_data.min()))
        maximum = float(max(chart_data.max()))
        figure.add_trace(
            go.Scatter(x=[minimum, maximum], y=[minimum, maximum], mode="lines", name="Ideal")
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        chart_data["Residual"] = chart_data["Actual"] - chart_data["Predicted"]
        figure = px.scatter(
            chart_data,
            x="Predicted",
            y="Residual",
            title="Residuals versus predicted",
            color_discrete_sequence=["#8d69e7"],
        )
        figure.add_hline(y=0, line_dash="dash")
        st.plotly_chart(figure, width="stretch")


def render_projection(result: dict[str, Any]) -> None:
    model = result["model"]
    preprocessor = model.named_steps["preprocessing"]
    x_train = result["x_train"]
    x_test = result["x_test"]
    y_test = result["y_test"]
    if len(x_train) > 10_000:
        sample_indices = x_train.sample(10_000, random_state=42).index
        x_train = x_train.loc[sample_indices]
    if len(x_test) > 5_000:
        sample_indices = x_test.sample(5_000, random_state=42).index
        x_test = x_test.loc[sample_indices]
        y_test = y_test.loc[sample_indices]

    transformed_train = preprocessor.transform(x_train)
    transformed_test = preprocessor.transform(x_test)
    feature_count = int(transformed_test.shape[1])
    color_values = y_test.astype(str) if result["task"] == "Classification" else y_test

    if feature_count == 1:
        values = (
            transformed_test.toarray().ravel()
            if sparse.issparse(transformed_test)
            else np.asarray(transformed_test).ravel()
        )
        frame = pd.DataFrame({"Feature 1": values, "Target": color_values.to_numpy()})
        figure = px.histogram(
            frame,
            x="Feature 1",
            color="Target",
            barmode="overlay",
            opacity=0.72,
            title="Distribution of the transformed feature",
        )
        caption = "One transformed feature is shown as a distribution."
    elif feature_count == 2:
        values = (
            transformed_test.toarray()
            if sparse.issparse(transformed_test)
            else np.asarray(transformed_test)
        )
        frame = pd.DataFrame(values, columns=["Feature 1", "Feature 2"])
        frame["Target"] = color_values.to_numpy()
        figure = px.scatter(
            frame,
            x="Feature 1",
            y="Feature 2",
            color="Target",
            title="Direct two-feature projection",
            opacity=0.75,
        )
        caption = "Two transformed features are plotted directly without dimensionality reduction."
    else:
        if sparse.issparse(transformed_train):
            reducer: PCA | TruncatedSVD = TruncatedSVD(n_components=2, random_state=42)
            method = "TruncatedSVD"
        else:
            reducer = PCA(n_components=2, random_state=42)
            method = "PCA"
        reducer.fit(transformed_train)
        projected = reducer.transform(transformed_test)
        variance = float(np.sum(reducer.explained_variance_ratio_))
        frame = pd.DataFrame(projected, columns=["Component 1", "Component 2"])
        frame["Target"] = color_values.to_numpy()
        figure = px.scatter(
            frame,
            x="Component 1",
            y="Component 2",
            color="Target",
            title=f"{method} projection of {feature_count} transformed features",
            opacity=0.75,
        )
        caption = (
            f"{method} was fitted on transformed training rows only. The first two components "
            f"explain {variance:.1%} of transformed variance."
        )
    figure.update_traces(marker={"size": 8})
    st.plotly_chart(figure, width="stretch")
    st.caption(caption)


def render_visualize() -> None:
    result = st.session_state.get("experiment_result")
    st.header("Visualize the result")
    st.markdown(
        '<p class="workbench-section-copy">Read the complete performance summary before interpreting projections and model plots.</p>',
        unsafe_allow_html=True,
    )
    if result is None:
        st.info("Run an experiment in the Model step to create visualizations.")
        st.button(
            "Go to model fitting",
            type="primary",
            on_click=set_pending_step,
            args=(WORKFLOW_STEPS[1],),
        )
        return

    with st.container(border=True):
        st.markdown(
            f"<span class='workbench-pill'>{result['experiment_mode']} · {result['tuning_mode']}</span>",
            unsafe_allow_html=True,
        )
        st.subheader(f"Best result: {result['model_name']}")
        st.caption(
            f"Ranked by {result['primary_metric']} · Target: {result['target']} · "
            f"Test set: {result['config']['test_percentage']}%"
        )
        metric_cards(result["metrics"])

    comparison = result["comparison"]
    if len(comparison) > 1:
        with st.container(border=True):
            st.subheader("Model leaderboard")
            display_columns = [
                column for column in comparison.columns if column not in {"Ranking score"}
            ]
            st.dataframe(
                comparison[display_columns],
                width="stretch",
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(format="%.4f")
                    for column in comparison.select_dtypes(include="number").columns
                    if column != "Rank"
                },
            )
            chart = px.bar(
                comparison,
                x="Model",
                y="Ranking score",
                color="Model",
                text_auto=".4f",
                title=f"Comparison by {result['primary_metric']}",
                color_discrete_sequence=["#5268f5", "#129b88", "#8d69e7", "#f0a35b"],
            )
            chart.update_layout(showlegend=False)
            st.plotly_chart(chart, width="stretch")

    with st.container(border=True):
        st.subheader("Detailed evaluation")
        if result["task"] == "Classification":
            render_classification_results(result)
        else:
            render_regression_results(result)

    with st.container(border=True):
        st.subheader("Feature-space projection")
        st.caption(
            "One feature uses a distribution, two features use a direct scatter plot, and higher-dimensional "
            "transformed data uses PCA or TruncatedSVD."
        )
        try:
            render_projection(result)
        except (ValueError, TypeError, MemoryError) as error:
            st.info(f"The projection could not be generated: {error}")

    with st.container(border=True):
        st.subheader("Best hyperparameters")
        st.json(result["best_parameters"])
        if result["failures"]:
            with st.expander(f"{len(result['failures'])} model warning(s)"):
                for failure in result["failures"]:
                    st.write(f"- {failure}")

        model_buffer = io.BytesIO()
        joblib.dump(result["model"], model_buffer)
        report = {
            "task": result["task"],
            "target": result["target"],
            "model": result["model_name"],
            "experiment_mode": result["experiment_mode"],
            "tuning_mode": result["tuning_mode"],
            "primary_metric": result["primary_metric"],
            "metrics": result["metrics"],
            "best_hyperparameters": result["best_parameters"],
            "preparation": result["config"],
        }
        download_a, download_b, download_c, download_d = st.columns(4)
        download_a.download_button(
            "Predictions",
            result["predictions_frame"].to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
            width="stretch",
        )
        download_b.download_button(
            "Leaderboard",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
            width="stretch",
        )
        download_c.download_button(
            "Full report",
            json.dumps(report, indent=2, default=str),
            file_name="model_report.json",
            mime="application/json",
            width="stretch",
        )
        download_d.download_button(
            "Best model",
            model_buffer.getvalue(),
            file_name="best_model.joblib",
            mime="application/octet-stream",
            width="stretch",
        )


def main() -> None:
    initialize_state()
    apply_theme()
    render_header()
    step = st.segmented_control(
        "Project workflow",
        WORKFLOW_STEPS,
        key="workflow_step",
        label_visibility="collapsed",
    )
    if step == WORKFLOW_STEPS[0]:
        render_prepare()
        return

    raw_data = read_stored_dataframe()
    config = st.session_state.get("prepared_config")
    if raw_data is None or config is None:
        st.warning("Complete and save the preparation recipe before fitting a model.")
        st.button(
            "Go to preparation",
            type="primary",
            on_click=set_pending_step,
            args=(WORKFLOW_STEPS[0],),
        )
        return

    if step == WORKFLOW_STEPS[1]:
        render_model(raw_data, config)
    else:
        render_visualize()


if __name__ == "__main__":
    main()

# ML Workbench

ML Workbench is a cloud-first Streamlit application for preparing data, fitting supervised machine-learning models, and understanding their results without writing code. The interface follows three spacious steps: **Prepare**, **Model**, and **Visualize**.

## Live application

[Open ML Workbench](https://navdeep-ml-workbench.streamlit.app/)

## Current capabilities

- CSV files only.
- Maximum upload size: 25 MB.
- Maximum shape: 100,000 rows and 200 columns.
- Classification and regression are supported.
- A complete per-feature preparation recipe is shown before training.
- Numeric-like text columns can be converted explicitly.
- Missing numeric features support `SimpleImputer`, `KNNImputer`, and experimental `IterativeImputer`.
- Missing categorical features can use the most-frequent value or a dedicated `Missing` category.
- Missing-value statistics are learned from training rows only to prevent data leakage.
- Missing targets are excluded rather than guessed.
- Blank strings, surrounding spaces, infinite values, and exact duplicate rows can be normalized.
- Constant and high-cardinality text columns are flagged and can be excluded.
- Numeric scaling supports automatic selection, no scaling, `StandardScaler`, `RobustScaler`, and `MinMaxScaler`.
- The test split is configurable from 10% to 40% and defaults to 20%.
- Classification uses a stratified split.
- Categorical features are one-hot encoded.
- Users can train one model or compare several models.
- Hyperparameters can be configured manually for every selected model.
- `RandomizedSearchCV` and bounded `GridSearchCV` searches are available with 3, 5, or 10 folds.
- Model selection uses cross-validation while the protected test set is reserved for final evaluation.
- Classification results include task metrics, a confusion matrix, a per-class report, ROC and precision–recall curves.
- Regression results include MAE, MSE, RMSE, R², actual-versus-predicted, and residual plots.
- One transformed feature uses a distribution plot, two use a direct scatter plot, and higher dimensions use PCA or TruncatedSVD.
- Predictions, leaderboard, JSON report, and the best fitted pipeline can be downloaded.

Date parsing, time-series-specific workflows, outlier guidance, class-imbalance resampling, and richer feature selection are planned extensions.

## Application workflow

1. **Prepare** — upload data, define the target and task, choose deterministic cleaning, configure imputation and scaling, and review every feature.
2. **Model** — choose single or comparison mode, then manual hyperparameters or cross-validated automatic search.
3. **Visualize** — inspect metrics before graphs, review detailed evaluation plots, explore the feature-space projection, and download outputs.

## Develop entirely in GitHub Codespaces

No local Python installation is required.

1. Push this repository to GitHub.
2. On the GitHub repository page, select **Code -> Codespaces -> Create codespace on main**.
3. Wait for the container to finish installing `requirements.txt`.
4. In the Codespaces terminal, run:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the forwarded port `8501` when Codespaces prompts you.
6. Commit and push changes from Codespaces.

The included `.devcontainer/devcontainer.json` uses Python 3.12, installs the dependencies, and forwards the Streamlit port automatically.

## Deploy to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) and connect GitHub.
2. Select **Create app**.
3. Choose this repository and its deployment branch.
4. Set the entrypoint to `app.py`.
5. In **Advanced settings**, select Python 3.12.
6. Deploy the application.

Community Cloud installs `requirements.txt`, reads `.streamlit/config.toml`, and rebuilds the app after new commits are pushed to the selected branch.

## Verify in Codespaces

```bash
python -m pytest
python -m streamlit run app.py
```

## Project structure

```text
.
|-- .devcontainer/          # GitHub Codespaces environment
|-- .streamlit/             # Cloud/runtime configuration
|-- ml_core/                # Validation, preprocessing, models, and metrics
|-- tests/                  # Automated tests
|-- app.py                  # Streamlit interface
|-- requirements.txt        # Cloud Python dependencies
`-- README.md               # Setup and deployment guide
```

## Privacy note

Uploaded datasets are processed in the running Streamlit session. Users should not upload confidential or personally identifiable data to a public demonstration deployment. Production use will require a separate privacy, retention, authentication, and resource-isolation design.

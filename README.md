# ML Workbench

ML Workbench is a cloud-first Streamlit application for training and evaluating supervised machine-learning models without writing code. Users upload a clean CSV dataset, choose classification or regression, select a target and model, configure model-specific hyperparameters, and download predictions, a model report, or the fitted pipeline.

## Current MVP rules

- CSV files only.
- Data must not contain missing values.
- Maximum upload size: 25 MB.
- Maximum shape: 100,000 rows and 200 columns.
- Classification and regression are supported.
- The test split is configurable from 10% to 40% and defaults to 20%.
- Classification uses a stratified split.
- Categorical features are one-hot encoded.
- Scaling is applied automatically to scale-sensitive algorithms.
- Hyperparameters are configured manually for one model at a time.

Automatic cross-validated hyperparameter search and missing-value handling are planned extensions.

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

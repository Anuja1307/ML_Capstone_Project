# Machine Learning Capstone Prediction System

An interactive Streamlit application for the **23CSE301 Machine Learning** capstone project. It provides inference interfaces for the completed Ames Housing regression and Telco Customer Churn classification tracks.

## Features

- Ames Housing sale-price prediction using the deployed Lasso Regression pipeline
- Telco customer-churn prediction using five classification algorithms:
  - Logistic Regression
  - Support Vector Classifier
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
  - Decision Tree Classifier
- Model-information tables based on the project evaluation results
- A responsive, accessible Streamlit interface

## Project structure

```text
app/
  app.py                  Streamlit interface
  model_loader.py         Metadata loaders
  pipeline_utils.py       Direct pipeline inference helpers
models/                   Serialized models and metadata
data/                     Ames Housing and Telco datasets
notebooks/                Regression, classification, and clustering notebooks
scripts/export_models.py  Model-export utility
```

## Requirements

- Python 3.10 or later
- Project packages from `requirements.txt`

Install the project packages in the Python environment you will use to run the app:

```powershell
python -m pip install -r requirements.txt
```

## Run the application

From the project root, use either launcher:

```powershell
.\run_app.ps1
```

```bat
run_app.bat
```

Or start Streamlit directly:

```powershell
python -m streamlit run app/app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Available pages

- **Home** — project overview
- **Regression Prediction** — Ames Housing sale-price estimation
- **Classification Prediction** — Telco customer-churn evaluation
- **Model Information** — datasets, preprocessing summaries, and evaluation tables

## Data and model artifacts

The repository includes the datasets, model metadata, and serialized model artifacts needed by the application. The Streamlit UI does not retrain models; it uses the existing artifacts in `models/`.

## Notes

- Keep the directory layout intact; the app resolves model paths relative to the project root.
- Prediction loads the fitted Joblib pipelines directly from `models/`; no secondary Python interpreter is started.

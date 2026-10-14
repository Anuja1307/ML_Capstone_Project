# Machine Learning Capstone

An interactive Streamlit application for **23CSE301 Machine Learning, Academic Year 2026–27**. The single application provides Ames Housing regression, Telco Customer Churn classification, and Airbnb NYC listing clustering.

## Project structure

```text
app/app.py                  Streamlit pages and navigation
app/model_loader.py         Cached model metadata loading
app/pipeline_utils.py       Regression and classification inference
models/                     Serialized estimators, preprocessing and metadata
data/                       Ames, Telco churn, and Airbnb NYC datasets
notebooks/                  Completed analysis notebooks for all tracks
scripts/export_models.py    Reproducible artifact generation
requirements.txt            Runtime dependencies matching exported artifacts
```

## Application modules

- **Regression Prediction:** Ames Housing sale-price estimate using the selected Lasso Regression pipeline. Its test-set comparison includes Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Decision Tree, Random Forest, Gradient Boosting, SVR, and KNN Regression.
- **Classification Prediction:** Ten Telco churn baseline algorithms: Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, Support Vector Classifier, Random Forest, AdaBoost, Gradient Boosting, Bagging, and MLP. Fitted preprocessing includes training-derived IQR capping, `TotalServices` feature engineering, one-hot encoding, and standard scaling. The notebook's tuned MLP experiment is reported separately from baseline comparison; Logistic Regression is its selected general-purpose baseline.
- **Clustering Analysis:** K-Means and Ward Agglomerative Hierarchical Clustering on the Airbnb NYC dataset. The notebook uses eight numeric and two categorical fields, fills missing `reviews_per_month` with zero, applies `log1p` to six skewed fields, standardizes numeric data and one-hot encodes categories. Both final models use the same reproducible 3,000-listing sample and three clusters. The page displays cluster metrics, PCA views, cluster sizes and original-unit profiles. New listings can be assigned with K-Means; agglomerative clustering has no out-of-sample `predict` method.
- **Model Information:** Notebook comparison metrics and selected configurations across all three tracks.

Clustering is unsupervised: cluster numbers are arbitrary identifiers, not ranked labels. PCA coordinates are for visualization; model fitting uses the full processed feature space. Clustering metrics on the Model Information page and analysis page use the notebook's same fixed sample and methodology.

## Setup and run locally

Use Python 3.13 and the pinned dependencies below. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/export_models.py
python -m streamlit run app/app.py
```

The artifact command recreates the notebook's train/test split and preprocessing for regression and classification and creates the clustering artifacts from the included dataset. It does not retrain on app interactions. Export can take a few minutes because it fits the requested estimators and the hierarchical model.

## Streamlit Community Cloud

1. Push this repository to GitHub, including the root `requirements.txt`, `app/`, `models/`, and `data/` directories. The bundled datasets and model artifacts are all small enough for ordinary Git hosting; the Random Forest artifact is the largest at about 18 MB.
2. In Streamlit Community Cloud, create an app from the repository and select `app/app.py` as the **Main file path**. Use Python 3.13 to match the serialized artifact runtime.
3. Deploy. No secrets or external data downloads are needed.

The repository includes trained runtime artifacts so deployment does not run model training. If artifacts are regenerated, commit the resulting files in `models/` with the matching pinned runtime dependencies. Do not commit credentials or secrets.

## Verification

Run `python -m streamlit run app/app.py` from the root. The export script reports held-out regression and classification metrics and writes all model artifacts. Artifacts are resolved relative to the repository, so the same layout works locally and in Community Cloud.

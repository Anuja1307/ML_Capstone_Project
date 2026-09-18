"""
scripts/export_models.py

Reproducible training and export of model pipelines and metadata
for the 23CSE301 Machine Learning Capstone Project.
Faithfully follows notebooks/regression.ipynb and notebooks/Classification.ipynb.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def export_regression():
    print("=" * 60)
    print("TRAINING & EXPORTING REGRESSION MODEL (Ames Housing)")
    print("=" * 60)

    data_path = DATA_DIR / "Ameshousing.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing regression dataset at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded Ames Housing dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Data cleaning strictly following notebooks/regression.ipynb
    NONE_MEANS_ABSENT = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'Electrical'
    ]
    ZERO_MEANS_ABSENT = ['GarageYrBlt', 'MasVnrArea']

    for c in NONE_MEANS_ABSENT:
        if c in df.columns:
            df[c] = df[c].fillna('None')
    for c in ZERO_MEANS_ABSENT:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Neighborhood lot frontage median
    neighborhood_medians = df.groupby('Neighborhood')['LotFrontage'].median().to_dict()
    global_lot_median = float(df['LotFrontage'].median())

    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda s: s.fillna(s.median()))
    df['LotFrontage'] = df['LotFrontage'].fillna(global_lot_median)

    # Filter partial-sale outliers (GrLivArea > 4000 & SalePrice < 300000)
    n0 = len(df)
    df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))].reset_index(drop=True)
    print(f"Outliers removed: {n0 - len(df)}")

    # Feature engineering strictly following notebooks/regression.ipynb
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

    X = df.drop(columns=['SalePrice', 'Id'])
    y = np.log1p(df['SalePrice'])

    strata = pd.qcut(df['SalePrice'], q=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strata
    )
    print(f"Train split: {X_train.shape}, Test split: {X_test.shape}")

    num_f = X.select_dtypes(include=np.number).columns.tolist()
    cat_f = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_f),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_f)
    ])

    # Best-performing validated regression model: Lasso Regression (alpha=0.001)
    lasso_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', Lasso(alpha=0.001, random_state=42))
    ])

    lasso_pipe.fit(X_train, y_train)

    pred = np.expm1(lasso_pipe.predict(X_test))
    actual = np.expm1(y_test)

    r2 = round(float(r2_score(actual, pred)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(actual, pred))))
    mae = round(float(mean_absolute_error(actual, pred)))

    print(f"Lasso Regression Evaluation -> R2: {r2} (Expected: 0.9385), RMSE: ${rmse} (Expected: 16622), MAE: ${mae} (Expected: 11812)")
    assert r2 == 0.9385, f"R2 mismatch: expected 0.9385, got {r2}"

    # Serialize trained regression pipeline
    model_path = MODELS_DIR / "regression_model.pkl"
    joblib.dump(lasso_pipe, model_path)
    print(f"Saved: {model_path}")

    # Build default feature vector (median for numeric, mode for categoricals)
    feature_defaults = {}
    for col in num_f:
        val = X_train[col].median()
        feature_defaults[col] = int(val) if val.is_integer() else round(float(val), 2)
    for col in cat_f:
        feature_defaults[col] = str(X_train[col].mode()[0])

    # Distinct categorical options
    cat_options = {col: sorted(X_train[col].dropna().unique().tolist()) for col in cat_f}

    metadata = {
        "dataset": "Ames Housing",
        "target": "SalePrice",
        "currency": "USD ($)",
        "selected_model": "Lasso Regression",
        "hyperparameters": {"alpha": 0.001, "random_state": 42},
        "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae},
        "all_features": X.columns.tolist(),
        "num_features": num_f,
        "cat_features": cat_f,
        "feature_defaults": feature_defaults,
        "cat_options": cat_options,
        "global_lot_median": global_lot_median,
        "neighborhood_medians": neighborhood_medians
    }

    meta_path = MODELS_DIR / "regression_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")


def export_classification():
    print("\n" + "=" * 60)
    print("TRAINING & EXPORTING CLASSIFICATION MODELS (Telco Churn Part A)")
    print("=" * 60)

    data_path = DATA_DIR / "TelcoCustomerChurn.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing classification dataset at {data_path}")

    data = pd.read_csv(data_path)
    print(f"Loaded Telco Customer Churn dataset: {data.shape[0]} rows, {data.shape[1]} columns")

    # Data cleaning strictly following notebooks/Classification.ipynb
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    # Dropping 11 missing rows as documented in Cell 40 & 41 of Classification.ipynb
    data = data.dropna(subset=['TotalCharges']).reset_index(drop=True)
    data = data.drop(columns=['customerID'])

    X = data.drop(columns=['Churn'])
    y = data['Churn'].map({'No': 0, 'Yes': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {X_train.shape}, Test split: {X_test.shape}")

    # Training-derived outlier bounds
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    outlier_bounds = {}
    for col in numerical_cols:
        Q1 = float(X_train[col].quantile(0.25))
        Q3 = float(X_train[col].quantile(0.75))
        IQR = Q3 - Q1
        lower_bound = float(Q1 - 1.5 * IQR)
        upper_bound = float(Q3 + 1.5 * IQR)
        outlier_bounds[col] = (lower_bound, upper_bound)
        X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
        X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)

    # Feature engineering: TotalServices
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for dataset in [X_train, X_test]:
        dataset['TotalServices'] = dataset[service_cols].apply(
            lambda row: (row == 'Yes').sum(), axis=1
        )

    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    models = {
        "logistic": ("Logistic Regression", LogisticRegression(random_state=42, max_iter=1000)),
        "svc": ("Support Vector Classifier", SVC(random_state=42)),
        "knn": ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        "dt": ("Decision Tree Classifier", DecisionTreeClassifier(random_state=42)),
        "nb": ("Gaussian Naive Bayes", GaussianNB())
    }

    metrics_table = []

    for key, (algo_name, estimator) in models.items():
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', estimator)
        ])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = round(float(accuracy_score(y_test, preds)), 6)
        f1 = round(float(f1_score(y_test, preds, average='weighted')), 6)

        metrics_table.append({
            "key": key,
            "Algorithm": algo_name,
            "Accuracy": acc,
            "Weighted F1": f1
        })
        print(f"Trained {algo_name:<28} | Acc: {acc:.4f} | F1: {f1:.4f}")

        model_path = MODELS_DIR / f"classification_{key}.pkl"
        joblib.dump(pipe, model_path)
        print(f"  -> Saved {model_path}")

    # Feature options and defaults for GUI widgets
    cat_options = {col: sorted(X_train[col].unique().tolist()) for col in categorical_features}
    feature_defaults = {}
    for col in numerical_features:
        if col != 'TotalServices':
            val = float(X_train[col].median())
            feature_defaults[col] = int(val) if val.is_integer() else round(val, 2)
    for col in categorical_features:
        feature_defaults[col] = str(X_train[col].mode()[0])

    metadata = {
        "dataset": "Telco Customer Churn",
        "target": "Churn",
        "classes": ["No", "Yes"],
        "outlier_bounds": outlier_bounds,
        "service_cols": service_cols,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "cat_options": cat_options,
        "feature_defaults": feature_defaults,
        "metrics_summary": metrics_table
    }

    meta_path = MODELS_DIR / "classification_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    export_regression()
    export_classification()
    print("\nAll models and metadata successfully exported!")

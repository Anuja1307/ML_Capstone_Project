# ML_Capstone_Project

## Machine Learning Capstone Project (23CSE301)
**Academic Year:** 2026–2027  
**Program:** B.Tech. Computer Science and Engineering  

---

## 📌 Project Overview
This repository contains the end-to-end Machine Learning pipeline developed for the **23CSE301 Machine Learning Capstone Project**. The project addresses three core machine learning tracks across two formal review phases:

| Track | Problem Statement & Dataset | Review Phase | Current Status |
| :--- | :--- | :---: | :---: |
| **Regression** | Ames Housing Sale Price Prediction (`data/Ameshousing.csv`) | Review 1 | **Completed** |
| **Classification (Part A)** | Telco Customer Churn – 5 Baseline Algorithms (`data/TelcoCustomerChurn.csv`) | Review 1 | **Completed** |
| **Classification (Part B)** | Telco Customer Churn – Ensembles & Neural Network | Review 2 | *Planned for Review 2* |
| **Clustering** | Customer Segmentation / Unsupervised Track (`data/clusteringdataset.csv`) | Review 2 | *Planned for Review 2* |

---

## 🏗️ Repository Structure

```
ML_Capstone_Project/
├── README.md                      # Project documentation and instructions
├── requirements.txt               # All Python dependencies with version specifications
├── data/                          # Raw datasets
│   ├── Ameshousing.csv            # Ames Housing regression dataset (1460 rows, 81 cols)
│   ├── TelcoCustomerChurn.csv     # Telco customer churn dataset (7043 rows, 21 cols)
│   └── clusteringdataset.csv      # Unsupervised clustering dataset (for Review 2)
├── notebooks/                     # Analytical and exploratory notebooks
│   ├── regression.ipynb           # Regression track: EDA, preprocessing, 10 models
│   ├── Classification.ipynb       # Classification Part A: EDA, preprocessing, 5 baseline models
│   └── clustering.ipynb           # Clustering track (scaffold for Review 2)
├── scripts/
│   └── export_models.py           # Reproducible model training and artifact export script
├── models/                        # Serialized model pipelines and metadata (.pkl via joblib)
│   ├── regression_model.pkl       # Validated Lasso Regression pipeline (R² = 0.9385)
│   ├── regression_metadata.json   # Ames feature schema, defaults, and validation metrics
│   ├── classification_logistic.pkl# Logistic Regression pipeline (Acc: 80.53%)
│   ├── classification_svc.pkl     # Support Vector Classifier pipeline (Acc: 79.25%)
│   ├── classification_knn.pkl     # K-Nearest Neighbors pipeline (Acc: 76.19%)
│   ├── classification_dt.pkl      # Decision Tree Classifier pipeline (Acc: 73.92%)
│   ├── classification_nb.pkl      # Gaussian Naive Bayes pipeline (Acc: 68.37%)
│   └── classification_metadata.json# Outlier bounds, service columns, and test benchmarks
└── app/                           # Optional Bonus GUI (Streamlit Web Interface)
    ├── __init__.py                # Package marker
    ├── app.py                     # Main Streamlit multi-page application
    ├── model_loader.py            # Cached artifact loading (@st.cache_resource)
    └── pipeline_utils.py          # Preprocessing, fixed-bound clipping, and inference utilities
```

---

## 💻 Interactive GUI (+1 Bonus Component)

As permitted under the **23CSE301 Capstone Guidelines (Bonus Opportunities: +1 Interactive GUI)**, an interactive web-based interface built with **Streamlit** is provided under `app/`.

### Key Features
- **Strictly Grounded in Completed Work:** Predictions use the actual trained scikit-learn pipelines saved via `joblib`. No hardcoded, synthetic, or random outputs.
- **Data-Leakage Protected:** Input scaling and one-hot encoding use pre-fitted encoders/scalers derived solely from training data.
- **Fixed-Bound Outlier Handling:** Outlier clipping for numerical features strictly applies fixed training-derived interquartile range (IQR) boundaries.
- **Automated Feature Engineering:** 
  - **Ames Housing:** Computes `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`, `HouseAge = YrSold - YearBuilt`, and weighted `TotalBath` automatically.
  - **Telco Churn:** Computes `TotalServices` dynamically by aggregating active subscriptions across 8 service categories.
- **Original Target Units:** Regression outputs are reported in original US Dollars (`$USD`) without currency alterations.
- **Factual Model Presentation:** Classification Part A presents all 5 baseline models objectively according to their empirical test metrics without prematurely declaring a final model before Review 2.

> [!NOTE]
> Classification Part B (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP) and the Clustering track will be integrated into the GUI during Review 2 upon completion.

---

## ⚙️ Environment Setup & Execution

### 1. Prerequisites
- Python 3.10 to 3.12 (or Anaconda / Miniconda environment)
- Git

### 2. Install Dependencies
Clone the repository and install all required packages:

```bash
git clone https://github.com/Anuja1307/ML_Capstone_Project.git
cd ML_Capstone_Project
pip install -r requirements.txt
```

### 3. (Optional) Re-export Trained Models
To retrain and re-serialize the models from raw data following the exact notebook specifications:

```bash
python scripts/export_models.py
```

### 4. Run the Streamlit Application
Launch the interactive web application from the project root:

```bash
streamlit run app/app.py
```

Open your web browser at `http://localhost:8501`.

---

## 🔬 Prediction Pipeline Architecture

```
User Input (Web Widgets)
         │
         ▼
Construct Single-Row DataFrame
         │
         ▼
Feature Engineering
  ├── Regression: TotalSF, HouseAge, TotalBath
  └── Classification: TotalServices
         │
         ▼
Leakage-Safe Preprocessing (ColumnTransformer)
  ├── Numeric: Fixed Training Bounds Clipping + Pre-fitted StandardScaler
  └── Categorical: Pre-fitted OneHotEncoder(handle_unknown='ignore')
         │
         ▼
Trained Estimator Pipeline
  ├── Regression: Lasso(alpha=0.001) with expm1() inverse log transform
  └── Classification: Selected Part-A Classifier (predict / predict_proba)
         │
         ▼
Results Display (Price in USD / Churn Status & Probability Gauge)
```

---

## 📊 Summary of Model Performance (Review 1 Scope)

### Regression Track (Ames Housing Price Prediction)
*Evaluated on an 80/20 stratified test split:*
- **Selected Validated Model:** Lasso Regression ($\alpha = 0.001$)
- **Validation $R^2$ Score:** `0.9385`
- **Root Mean Squared Error (RMSE):** `$16,622`
- **Mean Absolute Error (MAE):** `$11,812`

### Classification Track – Part A (Telco Customer Churn)
*Evaluated on an 80/20 stratified test split:*
1. **Logistic Regression:** Accuracy: `80.53%` | Weighted F1: `0.8014`
2. **Support Vector Classifier (SVC):** Accuracy: `79.25%` | Weighted F1: `0.7831`
3. **K-Nearest Neighbors (KNN):** Accuracy: `76.19%` | Weighted F1: `0.7634`
4. **Decision Tree Classifier:** Accuracy: `73.92%` | Weighted F1: `0.7406`
5. **Gaussian Naive Bayes:** Accuracy: `68.37%` | Weighted F1: `0.7021`

---

## 📜 Academic Integrity & Generative AI Citation
Generative AI assistance was utilized solely for code scaffolding, Streamlit user interface layout, and modular organization according to the Capstone Project Guidelines. All analytical interpretations, data preprocessing strategies, feature engineering logic, and algorithm evaluations stem directly from the completed coursework notebooks.

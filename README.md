# 23CSE301 Machine Learning — Capstone Project

End-to-end machine learning pipelines for three problem tracks — **Regression**, **Classification**, and **Clustering** — built with Python 3, scikit-learn, Pandas, NumPy, Matplotlib and Seaborn.

**Course:** 23CSE301 Machine Learning
**Programme:** B.Tech. Computer Science and Engineering, III Year
**Academic Year:** 2026–27
**Team size:** 3

Each track uses its own dataset and its own notebook. Every pipeline covers the full workflow: data loading and audit, exploratory data analysis, cleaning, feature engineering, encoding and scaling, model training, comparative evaluation, and visualisation.

---

## 1. Project Overview

| Track | Dataset | Problem type | Target | Notebook |
|---|---|---|---|---|
| Regression | Ames Housing | Supervised, continuous | `SalePrice` | `notebooks/regression.ipynb` |
| Classification | Telco Customer Churn | Supervised, binary | `Churn` | `notebooks/classification.ipynb` |
| Clustering | New York City Airbnb | Unsupervised | none (labels used only for post-hoc interpretation) | `notebooks/clustering.ipynb` |

Common conventions across all tracks:

- `random_state=42` everywhere a seed applies, so results are reproducible.
- A consistent **80:20 train/test split**, stratified, used by every algorithm within a track so metric comparisons are fair.
- All scalers and encoders are **fitted on the training set only** and then applied to both splits. In the regression track the preprocessor sits inside a `Pipeline`, which makes leakage structurally impossible even during cross-validation.
- Results are reported as a single comparison table per track, not as scattered print statements.

---

## 2. Dataset Overview

### 2.1 Ames Housing — Regression

Residential property sales in Ames, Iowa. Sourced from the Kaggle competition *House Prices — Advanced Regression Techniques*. Only the labelled training file is used, because the competition test file has no `SalePrice` values; the held-out test set is produced by an internal split instead.

| Property | Value |
|---|---|
| Rows × Columns | 1460 × 81 |
| Feature types | 43 object, 35 int64, 3 float64 |
| Target | `SalePrice` (continuous, USD) |
| Target range | \$34,900 – \$755,000 (median \$163,000, mean \$180,921) |
| Target skewness | 1.883 (right-skewed → `log1p` transform applied before modelling) |
| Duplicates | 0 |
| Missing values | 19 columns affected |

Notable characteristics:

- Missingness is **structural, not random**, in most columns. `PoolQC` (99.52% missing), `MiscFeature` (96.30%), `Alley` (93.77%) and `Fence` (80.75%) are near-empty, and the five Garage columns are each missing exactly the same 81 rows. These are houses with no pool, no alley and no garage — median-filling would invent features that do not exist. They are filled with `'None'` or `0`. `LotFrontage` is the one genuinely unknown measurement and is filled with the neighbourhood median.
- Strongest linear predictors: `OverallQual` (r = 0.79) and `GrLivArea` (r = 0.71).
- Strong multicollinearity between `GarageCars`–`GarageArea` (0.88), `GrLivArea`–`TotRmsAbvGrd` (0.83) and `TotalBsmtSF`–`1stFlrSF` (0.82). Each pair measures roughly the same thing twice, which inflates OLS coefficient variance and is the main reason regularised models were expected to perform well.

**Engineered features:** `TotalSF` (basement + 1st floor + 2nd floor, r = 0.833 — stronger than any original feature), `HouseAge` (`YrSold − YearBuilt`, r = −0.524), `TotalBath` (bathroom counts combined, half-baths weighted 0.5, r = 0.636).

**After preprocessing:** 2 partial-sale outliers removed (houses > 4000 sq ft that sold below \$300k), 0 missing values, 1166 train / 292 test rows, 82 input columns expanding to **302 features** after one-hot encoding.

### 2.2 Telco Customer Churn — Classification

Customer records from a fictional telecommunications company. The task is to predict whether a customer leaves the service, based on demographics, subscribed services, contract type and billing information.

| Property | Value |
|---|---|
| Rows × Columns | 7043 × 21 (20 input features + target) |
| Target | `Churn` (Yes / No, encoded as 1 / 0) |
| Class balance | 73.46% No, 26.54% Yes — imbalanced |
| Duplicates | 0 |
| Missing values | 11 hidden blanks in `TotalCharges` (~0.16%) |

Notable characteristics:

- `TotalCharges` is stored as an object because 11 records contain blank strings. These were coerced to `NaN` with `pd.to_numeric(errors='coerce')` and the 11 rows dropped, leaving **7032 rows**.
- `customerID` is dropped — a unique identifier carries no predictive information.
- Because the classes are imbalanced, accuracy alone is not sufficient. Precision, recall, weighted F1 and ROC-AUC are also reported.
- `tenure` and `TotalCharges` are strongly correlated, as expected — longer-staying customers accumulate more charges.
- Churned customers show **lower tenure** and **higher monthly charges** than retained customers.

**Outlier treatment:** IQR bounds computed on the **training split only**, then applied as capping (clipping) to both splits rather than deleting rows, so no customer record is lost.

**Engineered feature:** `TotalServices` — a count of how many of the eight service options a customer subscribes to, giving a single measure of service engagement.

**After preprocessing:** 5625 train / 1407 test rows (stratified), 19 input columns expanding to **46 features** after encoding.

### 2.3 New York City Airbnb — Clustering

Airbnb listings across the five boroughs of New York City, covering location, room type, price, availability and review activity. Used for the unsupervised track: K-Means and Agglomerative Hierarchical Clustering, evaluated with Silhouette Score, Davies–Bouldin Index and Calinski–Harabasz Index, and visualised with PCA (2 components) and t-SNE.

Ground-truth labels such as `neighbourhood_group` are used **only after fitting**, to interpret and validate the clusters — never during training.

*This track is scheduled for Review 2.*

---

## 3. Results — Regression Track (Ames Housing)

All ten algorithms were trained on the same preprocessed data and evaluated on the same held-out test split of 292 houses. The target was modelled in log space (`log1p`) and predictions were inverse-transformed with `expm1` before scoring, so R², RMSE and MAE are all reported in **dollars** and are directly comparable.

| # | Model | R² | RMSE (\$) | MAE (\$) |
|---|---|---|---|---|
| 1 | **Lasso Regression** | **0.9385** | **16,622** | **11,812** |
| 2 | ElasticNet | 0.9366 | 16,874 | 12,158 |
| 3 | Ridge Regression | 0.9328 | 17,380 | 12,592 |
| 4 | Gradient Boosting | 0.9188 | 19,102 | 13,528 |
| 5 | Linear Regression | 0.9111 | 19,985 | 14,395 |
| 6 | Random Forest | 0.8941 | 21,816 | 14,913 |
| 7 | SVR | 0.8896 | 22,272 | 14,795 |
| 8 | Polynomial Regression | 0.8687 | 24,287 | 17,392 |
| 9 | KNN Regressor | 0.8587 | 25,194 | 17,853 |
| 10 | Decision Tree | 0.7763 | 31,706 | 21,041 |

Model configurations: `Ridge(alpha=10)`, `Lasso(alpha=0.001)`, `ElasticNet(alpha=0.001, l1_ratio=0.5)`, `PolynomialFeatures(degree=2)` on six selected columns, `DecisionTreeRegressor(max_depth=8)`, `RandomForestRegressor(n_estimators=200)`, `SVR(C=10, kernel='rbf')`, `KNeighborsRegressor(n_neighbors=5)`.

---

## 4. Results — Classification Track, Part A (Telco Customer Churn)

Five Part-A algorithms evaluated on the same 1407-row held-out test set.

| # | Algorithm | Accuracy | Weighted F1 |
|---|---|---|---|
| 1 | **Logistic Regression** | **0.8053** | **0.8014** |
| 2 | Support Vector Classifier | 0.7925 | 0.7831 |
| 3 | K-Nearest Neighbors | 0.7612 | 0.7626 |
| 4 | Decision Tree Classifier | 0.7271 | 0.7278 |
| 5 | Gaussian Naive Bayes | 0.6837 | 0.7021 |

Confusion matrices (rows = actual, columns = predicted; order: No Churn, Churn):

| Algorithm | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 917 | 116 | 158 | 216 |
| Support Vector Classifier | 930 | 103 | 189 | 185 |
| K-Nearest Neighbors | 858 | 175 | 161 | 213 |
| Decision Tree Classifier | 838 | 195 | 189 | 185 |
| Gaussian Naive Bayes | 652 | 381 | 64 | 310 |

Observations:

- **Logistic Regression is the strongest Part-A model** on both accuracy and weighted F1, which is a reasonable outcome for a churn problem where much of the signal (contract type, tenure, monthly charges) is close to linearly separable after one-hot encoding.
- **SVC has the fewest false positives (103)** but misses more churners than Logistic Regression (189 vs 158). If the business cost of failing to catch a churner exceeds the cost of a wasted retention offer, SVC's higher precision is the less useful trade.
- **Gaussian Naive Bayes ranks last on accuracy but catches the most churners (310 of 374 true positives, recall ≈ 0.83).** It buys that recall with 381 false positives. Its conditional-independence assumption is clearly violated here — `tenure`, `TotalCharges` and `MonthlyCharges` are related — which explains the weak overall performance, but the recall behaviour is worth noting rather than dismissing.
- The untuned Decision Tree overfits, as expected from an unrestricted tree on 46 features.

Part B (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP) and the consolidated 10-algorithm table with Precision, Recall and ROC-AUC are due in Review 2.

---

## 5. Conclusion — Regression Track

**Lasso Regression is the best-performing model on the Ames Housing dataset**, with an R² of **0.9385**, an RMSE of **\$16,622** and an MAE of **\$11,812** on the held-out test set. On a dataset whose median sale price is \$163,000, a typical absolute error of roughly \$11,800 is about 7% of the median price.

Why Lasso wins here follows directly from the structure of the data:

1. **The feature space is wide relative to the sample.** One-hot encoding expands 82 input columns into 302 features, trained on only 1166 rows. Unregularised ordinary least squares has too much freedom in that setting, and Linear Regression duly finishes fifth (R² 0.9111) — well behind all three regularised variants.

2. **The strongest predictors are collinear.** `GarageCars`/`GarageArea`, `GrLivArea`/`TotRmsAbvGrd` and `TotalBsmtSF`/`1stFlrSF` each measure the same underlying quantity twice. Collinearity inflates coefficient variance in OLS. The L1 penalty resolves this by driving one member of each redundant pair toward zero, producing a sparse model that keeps the informative feature and discards its duplicate.

3. **Feature selection beats feature averaging on this data.** Ridge (R² 0.9328) shrinks all 302 coefficients but eliminates none, so it still carries the noise from the many near-empty categorical dummies. ElasticNet (R² 0.9366) sits between the two, as its mixed penalty would predict. The ordering Lasso > ElasticNet > Ridge > Linear is exactly what the collinearity diagnosis anticipated.

4. **Linear structure dominates after the log transform.** `log1p(SalePrice)` corrects the 1.883 skew and linearises the price–area relationship, so the tree ensembles gain nothing from modelling non-linear interactions. Gradient Boosting (0.9188) and Random Forest (0.8941) both finish behind the linear family, and the single Decision Tree finishes last (0.7763) because one tree cannot express a smooth continuous price surface without deep, high-variance splits.

The engineered `TotalSF` feature deserves part of the credit. At r = 0.833 with `SalePrice`, it correlates more strongly than `OverallQual` (0.79), the best original feature in the dataset, and gives the linear models a single clean signal for total usable space where the raw data offered three fragmented columns.

**Selected model: Lasso Regression (`alpha=0.001`).**

---

## 6. Setup Instructions

### Prerequisites

- Python 3.11 or newer (the notebooks were run on Python 3.13 and 3.14)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`scikit-learn >= 1.2` is required. The preprocessing pipelines use `OneHotEncoder(sparse_output=False)`, which does not exist in earlier releases and will raise a `TypeError`.

### 4. Add the datasets

Place the three CSV files in `data/` with these exact filenames:

| File | Source |
|---|---|
| `Ameshousing.csv` | Kaggle — *House Prices: Advanced Regression Techniques* (`train.csv`, renamed) |
| `TelcoCustomerChurn.csv` | Kaggle — *Telco Customer Churn* (IBM sample dataset) |
| `NYCAirbnb.csv` | Kaggle — *New York City Airbnb Open Data* |

### 5. Launch Jupyter and run the notebooks

```bash
cd notebooks
jupyter notebook
```

The notebooks read their data with relative paths (`../data/...`), so they must be launched from inside the `notebooks/` directory with the CSV files sitting in `data/` one level up.

Open a notebook and run **Cell → Run All**. Every notebook is written to execute top-to-bottom without errors. Approximate runtimes on a standard laptop: regression ~2–3 minutes (Random Forest with 200 estimators and SVR are the slowest cells), classification ~1–2 minutes.

Because `random_state=42` is set throughout, a clean run reproduces the exact metrics reported in Sections 3 and 4.

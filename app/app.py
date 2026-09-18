"""
app/app.py

Interactive Streamlit Application for 23CSE301 Machine Learning Capstone Project.
Supports:
  1. Regression Track (Ames Housing Price Prediction)
  2. Classification Track - Part A (Telco Customer Churn Prediction)
"""

import streamlit as st
import pandas as pd
try:
    from model_loader import (
        load_regression_metadata,
        load_classification_metadata,
    )
    from pipeline_utils import (
        predict_regression_price,
        predict_classification_churn,
    )
except ImportError:
    from app.model_loader import (
        load_regression_metadata,
        load_classification_metadata,
    )
    from app.pipeline_utils import (
        predict_regression_price,
        predict_classification_churn,
    )

# Page configuration
st.set_page_config(
    page_title="ML Capstone – Prediction System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom academic-style CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: min(20rem, 88vw);
    }
    [data-testid="stSidebar"] > div:first-child {
        width: min(20rem, 88vw);
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        justify-content: flex-start;
        min-height: 2.5rem;
        margin: 0.15rem 0;
        border: 1px solid transparent;
        border-radius: 0.35rem;
        text-align: left;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        background-color: #1E3A8A;
        border-color: #1E3A8A;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
        background-color: transparent;
        color: inherit;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:focus-visible {
        outline: 3px solid #60A5FA;
        outline-offset: 2px;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
        border: 1px solid #E5E7EB;
    }
    .result-price {
        font-size: 2.2rem;
        font-weight: 700;
        color: #047857;
    }
    .result-churn {
        font-size: 1.8rem;
        font-weight: 700;
        color: #B91C1C;
    }
    .result-stay {
        font-size: 1.8rem;
        font-weight: 700;
        color: #047857;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.title("ML CAPSTONE")
    st.sidebar.markdown("23CSE301 Machine Learning")
    st.sidebar.caption("Academic Year 2026–27")
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### NAVIGATION")

    page_names = [
        "Home",
        "Regression Prediction",
        "Classification Prediction",
        "Model Information",
    ]
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Home"

    def select_page(page_name):
        st.session_state.selected_page = page_name

    for page_name in page_names:
        st.sidebar.button(
            page_name,
            key=f"nav_{page_name.lower().replace(' ', '_')}",
            type="primary" if st.session_state.selected_page == page_name else "secondary",
            use_container_width=True,
            on_click=select_page,
            args=(page_name,),
        )

    return st.session_state.selected_page


def render_home():
    st.markdown(
        "<div class=\'main-header\'>Machine Learning Capstone \u2013 Prediction System</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class=\'sub-header\'>Interactive prediction interface for Regression and Classification Part-A tracks.</div>",
        unsafe_allow_html=True
    )

    st.subheader("Project Overview")
    st.markdown("""
    This machine learning system provides end-to-end predictive capabilities developed under the
    **23CSE301 Machine Learning Capstone Guidelines**. It demonstrates production-ready pipelines
    with strict data-leakage protection, validated feature engineering, and real-time model inference.
    """)

    st.markdown("### Project Tracks")

    with st.expander("Ames Housing Sale Price Prediction (Regression)", expanded=True):
        st.markdown(r"""
        - **Objective:** Predict residential property sale prices in Ames, Iowa based on architectural and structural features.
        - **Target Variable:** `SalePrice` (Continuous, evaluated in USD).
        - **Evaluated Model:** Lasso Regression ($\alpha = 0.001$) achieving $R^2 = 0.9385$, with logarithmic target transformation and standardized features.
        - **Feature Engineering:** `TotalSF` (total usable square footage), `HouseAge`, and weighted `TotalBath`.
        - Use **Regression Prediction** from the sidebar to make predictions.
        """)

    with st.expander("Telco Customer Churn Prediction (Classification Part A)", expanded=True):
        st.markdown("""
        - **Objective:** Identify telecom customers at risk of churning to support proactive retention strategies.
        - **Target Variable:** `Churn` (Binary: 0 = Stay, 1 = Churn).
        - **Evaluated Models:** 5 benchmark baseline algorithms (Logistic Regression, SVC, KNN, Decision Tree, Gaussian Naive Bayes).
        - **Feature Engineering:** `TotalServices` derived automatically from 8 distinct telecom subscription features.
        - **Leakage Prevention:** Training-only IQR outlier clipping, OneHotEncoding, and StandardScaling.
        - Use **Classification Prediction** from the sidebar to make predictions.
        """)

    st.markdown("---")
    st.subheader("Technical Stack")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("""
        **Interface**
        - Streamlit (Anaconda Python)
        - Pandas, NumPy
        """)
    with tc2:
        st.markdown("""
        **ML Libraries**
        - scikit-learn (Pipeline + ColumnTransformer)
        - Joblib (model serialization)
        """)
    with tc3:
        st.markdown("""
        **Visualization**
        - Matplotlib, Seaborn
        - Streamlit charts
        """)

def render_regression_page():
    st.markdown('<div class="main-header">Regression Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ames Housing: predict home sale prices in USD using the validated Lasso Regression pipeline.</div>',
        unsafe_allow_html=True
    )

    metadata = load_regression_metadata()
    if metadata is None:
        st.error("Regression metadata could not be loaded. Verify that models/regression_metadata.json exists.")
        return

    st.info(
        f"**Active Model:** {metadata.get('selected_model', 'Lasso Regression')} | "
        f"**Test Performance:** $R^2 = {metadata['metrics']['R2']:.4f}$ | "
        f"**RMSE:** ${metadata['metrics']['RMSE']:,} | "
        f"**MAE:** ${metadata['metrics']['MAE']:,}"
    )

    with st.form("regression_form"):
        st.subheader("Property Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7,
                                     help="Rates the overall material and finish of the house.")
            gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=5000, value=1700, step=50)
            total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3500, value=900, step=50)

        with c2:
            first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=300, max_value=4000, value=1000, step=50)
            second_flr_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0, max_value=2500, value=700, step=50)
            garage_cars = st.slider("Garage Car Capacity", min_value=0, max_value=4, value=2)

        with c3:
            garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=500, step=25)
            year_built = st.number_input("Year Built", min_value=1870, max_value=2010, value=2003)
            yr_sold = st.number_input("Year Sold", min_value=2006, max_value=2010, value=2008)

        st.subheader("Property Characteristics")
        c4, c5, c6 = st.columns(3)

        neighborhoods = metadata.get("cat_options", {}).get("Neighborhood", ["CollgCr", "NAmes", "Edwards"])
        with c4:
            neighborhood = st.selectbox("Neighborhood", options=neighborhoods,
                                        index=neighborhoods.index("CollgCr") if "CollgCr" in neighborhoods else 0)
        with c5:
            bldg_types = metadata.get("cat_options", {}).get("BldgType", ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"])
            bldg_type = st.selectbox("Building Type", options=bldg_types,
                                     index=bldg_types.index("1Fam") if "1Fam" in bldg_types else 0)
        with c6:
            house_styles = metadata.get("cat_options", {}).get("HouseStyle", ["1Story", "2Story", "1.5Fin", "SLvl"])
            house_style = st.selectbox("House Style", options=house_styles,
                                       index=house_styles.index("2Story") if "2Story" in house_styles else 0)

        st.subheader("Additional Details")
        c7, c8, c9, c10 = st.columns(4)
        with c7:
            full_bath = st.selectbox("Full Baths Above Grade", [1, 2, 3, 4], index=1)
        with c8:
            half_bath = st.selectbox("Half Baths Above Grade", [0, 1, 2], index=1)
        with c9:
            bsmt_full_bath = st.selectbox("Basement Full Baths", [0, 1, 2, 3], index=1)
        with c10:
            bsmt_half_bath = st.selectbox("Basement Half Baths", [0, 1, 2], index=0)

        # Derived Feature Preview
        calc_total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
        calc_house_age = max(0, yr_sold - year_built)
        calc_total_bath = full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath

        st.markdown(f"""
        > **Automatically Derived Features:**
        > - **Total Square Footage (`TotalSF`):** `{calc_total_sf:,} sq ft`
        > - **House Age at Sale (`HouseAge`):** `{calc_house_age} years`
        > - **Total Weighted Bathrooms (`TotalBath`):** `{calc_total_bath}`
        """)

        with st.expander("Advanced Architectural Attributes (Defaults Pre-filled)", expanded=False):
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                lot_frontage = st.number_input(
                    "Lot Frontage (ft, 0 = use neighborhood median)",
                    min_value=0, max_value=300, value=0
                )
                lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=9500, step=500)
                overall_cond = st.slider("Overall Condition (1-9)", min_value=1, max_value=9, value=5)
            with ac2:
                kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"], index=1)
                exter_qual = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa"], index=1)
                bsmt_qual = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "None"], index=1)
            with ac3:
                central_air = st.selectbox("Central Air Conditioning", ["Y", "N"], index=0)
                fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3], index=1)
                heating_qc = st.selectbox("Heating Quality & Condition", ["Ex", "Gd", "TA", "Fa", "Po"], index=0)

        submitted = st.form_submit_button("Predict House Price", use_container_width=True)

    if submitted:
        user_inputs = {
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "YearBuilt": year_built,
            "YrSold": yr_sold,
            "Neighborhood": neighborhood,
            "BldgType": bldg_type,
            "HouseStyle": house_style,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BsmtFullBath": bsmt_full_bath,
            "BsmtHalfBath": bsmt_half_bath,
            "LotArea": lot_area,
            "OverallCond": overall_cond,
            "KitchenQual": kitchen_qual,
            "ExterQual": exter_qual,
            "BsmtQual": bsmt_qual,
            "CentralAir": central_air,
            "Fireplaces": fireplaces,
            "HeatingQC": heating_qc,
        }
        if lot_frontage > 0:
            user_inputs["LotFrontage"] = float(lot_frontage)

        with st.spinner("Running prediction via Anaconda Python..."):
            try:
                predicted_price = predict_regression_price(user_inputs)
                st.markdown(f"""
                <div class="result-box" style="background-color: #ECFDF5;">
                    <div style="font-size: 1rem; color: #065F46; font-weight: 600;">ESTIMATED MARKET VALUE</div>
                    <div class="result-price">${predicted_price:,.2f} USD</div>
                    <div style="color: #047857; margin-top: 0.5rem; font-size: 0.95rem;">
                        <b>Algorithm Used:</b> Lasso Regression (L1-regularized linear model, &alpha;=0.001)<br>
                        <b>Target Schema:</b> Log-transformed <code>SalePrice</code> with inverse <code>expm1()</code> transformation.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Prediction Error: {exc}")

def render_classification_page():
    st.markdown('<div class="main-header">Telco Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Classification Track — Part A</div>',
        unsafe_allow_html=True
    )

    metadata = load_classification_metadata()
    if metadata is None:
        st.error("Missing classification metadata at models/classification_metadata.json.")
        return

    # Algorithm selector based on Part-A models
    algo_options = {
        "logistic": "Logistic Regression (Accuracy: 80.53%, F1: 0.8014)",
        "svc": "Support Vector Classifier (Accuracy: 79.25%, F1: 0.7831)",
        "knn": "K-Nearest Neighbors (Accuracy: 76.19%, F1: 0.7634)",
        "dt": "Decision Tree Classifier (Accuracy: 73.92%, F1: 0.7406)",
        "nb": "Gaussian Naive Bayes (Accuracy: 68.37%, F1: 0.7021)"
    }

    st.subheader("Model Configuration")
    selected_key = st.selectbox(
        "Select Classification Algorithm (Part A):",
        options=list(algo_options.keys()),
        format_func=lambda k: algo_options[k],
        index=0
    )

    with st.form("classification_form"):
        st.subheader("Customer Demographics")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        with d2:
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        with d3:
            partner = st.selectbox("Partner", ["Yes", "No"], index=1)
        with d4:
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)

        st.subheader("Account & Billing Information")
        b1, b2, b3 = st.columns(3)
        with b1:
            tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
        with b2:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                index=0
            )
        with b3:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=1.0)
            calc_est_total = round(tenure * monthly_charges, 2)
            total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(calc_est_total), step=10.0)

        st.subheader("Telecommunications & Internet Services")
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"]
            )

        with s2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
            internet_flag = internet_service != "No"
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"] if internet_flag else ["No internet service"]
            )

        with s3:
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"] if internet_flag else ["No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"] if internet_flag else ["No internet service"]
            )

        with s4:
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"] if internet_flag else ["No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"] if internet_flag else ["No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"] if internet_flag else ["No internet service"]
            )

        # Derived Feature Calculation
        service_fields = [
            phone_service, multiple_lines, online_security, online_backup,
            device_protection, tech_support, streaming_tv, streaming_movies
        ]
        calc_total_services = sum(1 for v in service_fields if v == "Yes")

        st.markdown(f"""
        > **Automatically Derived Feature:**
        > - **Total Subscribed Services (`TotalServices`):** `{calc_total_services} / 8 services`
        """)

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        user_inputs = {
            "gender": gender,
            "SeniorCitizen": int(senior_citizen),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        with st.spinner(f"Running {algo_options[selected_key].split('(')[0].strip()} via Anaconda Python..."):
            try:
                pred_class, prob_churn = predict_classification_churn(user_inputs, selected_key)

                if pred_class == 1:
                    st.markdown("""
                    <div class="result-box" style="background-color: #FEF2F2; border-color: #F87171;">
                        <div style="font-size: 0.95rem; color: #991B1B; font-weight: 600;">PREDICTION RESULT</div>
                        <div class="result-churn">Prediction: Likely to Churn</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-box" style="background-color: #ECFDF5; border-color: #34D399;">
                        <div style="font-size: 0.95rem; color: #065F46; font-weight: 600;">PREDICTION RESULT</div>
                        <div class="result-stay">Prediction: Likely to Stay</div>
                    </div>
                    """, unsafe_allow_html=True)

                if prob_churn is not None:
                    st.markdown(f"**Estimated Churn Probability:** `{prob_churn * 100:.2f}%`")
                    st.progress(prob_churn)
                else:
                    st.info(
                        "Standard Support Vector Classifier uses margin-based classification without Platt scaling probabilities. "
                        "Select Logistic Regression, KNN, Gaussian Naive Bayes, or Decision Tree to view calibrated probabilities."
                    )

            except Exception as exc:
                st.error(f"Prediction Error: {exc}")

def render_model_info_page():
    st.markdown('<div class="main-header">Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Factual performance summaries from the project notebooks.</div>',
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs([
        "Regression Track",
        "Classification Track — Part A",
    ])

    with tab1:
        st.subheader("Regression Track")
        st.markdown(r"""
        - **Dataset:** Ames Housing (`data/Ameshousing.csv`, 1,460 records).
        - **Target:** `SalePrice` (Transformed via `log1p` during training, evaluated in original USD units via `expm1`).
        - **Data Cleaning:** Structural absences in Garage/Pool/Basement columns replaced with `'None'`/`0`; neighborhood-median imputation for `LotFrontage`; two partial-sale outliers ($> 4000$ sq ft, $< \$300,000$) removed.
        - **Feature Engineering:**
          - `TotalSF`: `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` ($r = 0.833$ with `SalePrice`)
          - `HouseAge`: `YrSold - YearBuilt` ($r = -0.524$)
          - `TotalBath`: `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` ($r = 0.636$)
        """)

        st.markdown("#### Regression Algorithm Comparison (Test Set Split: 80/20 Stratified)")
        reg_data = [
            {"Rank": 1, "Model": "Lasso Regression (Deployed in GUI)", "R2": 0.9385, "RMSE ($)": 16622, "MAE ($)": 11812},
            {"Rank": 2, "Model": "ElasticNet", "R2": 0.9366, "RMSE ($)": 16874, "MAE ($)": 12158},
            {"Rank": 3, "Model": "Ridge Regression", "R2": 0.9328, "RMSE ($)": 17380, "MAE ($)": 12592},
            {"Rank": 4, "Model": "Gradient Boosting", "R2": 0.9188, "RMSE ($)": 19102, "MAE ($)": 13528},
            {"Rank": 5, "Model": "Linear Regression", "R2": 0.9111, "RMSE ($)": 19985, "MAE ($)": 14395},
            {"Rank": 6, "Model": "Random Forest", "R2": 0.8941, "RMSE ($)": 21816, "MAE ($)": 14913},
            {"Rank": 7, "Model": "SVR (Support Vector Regressor)", "R2": 0.8896, "RMSE ($)": 22272, "MAE ($)": 14795},
            {"Rank": 8, "Model": "Polynomial Regression (deg=2)", "R2": 0.8687, "RMSE ($)": 24287, "MAE ($)": 17392},
            {"Rank": 9, "Model": "KNN Regressor", "R2": 0.8587, "RMSE ($)": 25194, "MAE ($)": 17853},
            {"Rank": 10, "Model": "Decision Tree Regressor", "R2": 0.7763, "RMSE ($)": 31706, "MAE ($)": 21041},
        ]
        st.dataframe(pd.DataFrame(reg_data), hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Classification Track — Part A")
        st.markdown("""
        - **Dataset:** Telco Customer Churn (`data/TelcoCustomerChurn.csv`, 7,043 records).
        - **Target:** `Churn` (Mapped as `{'No': 0, 'Yes': 1}`).
        - **Data Cleaning:** Coerced `TotalCharges` to numeric, dropped the 11 blank records (0.16%), removed `customerID`.
        - **Leakage Prevention:**
          - Stratified 80/20 train-test split applied before preprocessing.
          - IQR outlier bounds calculated on `X_train` only (`tenure`: `[-61.5, 126.5]`, `MonthlyCharges`: `[-45.58, 171.43]`, `TotalCharges`: `[-4670.05, 8884.75]`).
          - Feature scaling and encoding fitted on training data only.
        - **Feature Engineering:** `TotalServices` dynamically derived as the sum of 8 active subscription services.
        """)

        st.markdown("#### Part-A Algorithm Comparison (Test Set Split: 80/20 Stratified)")
        clf_data = [
            {"Algorithm": "Logistic Regression", "Accuracy": "80.53%", "Weighted F1": "0.8014", "Type": "Linear Baseline"},
            {"Algorithm": "Support Vector Classifier", "Accuracy": "79.25%", "Weighted F1": "0.7831", "Type": "Kernel Margin Classifier"},
            {"Algorithm": "K-Nearest Neighbors", "Accuracy": "76.19%", "Weighted F1": "0.7634", "Type": "Instance-based (k=5)"},
            {"Algorithm": "Decision Tree Classifier", "Accuracy": "73.92%", "Weighted F1": "0.7406", "Type": "Tree Baseline"},
            {"Algorithm": "Gaussian Naive Bayes", "Accuracy": "68.37%", "Weighted F1": "0.7021", "Type": "Probabilistic Baseline"}
        ]
        st.dataframe(pd.DataFrame(clf_data), hide_index=True, use_container_width=True)

def main():
    selected_page = render_sidebar()
    if selected_page == "Home":
        render_home()
    elif selected_page == "Regression Prediction":
        render_regression_page()
    elif selected_page == "Classification Prediction":
        render_classification_page()
    elif selected_page == "Model Information":
        render_model_info_page()


if __name__ == "__main__":
    main()

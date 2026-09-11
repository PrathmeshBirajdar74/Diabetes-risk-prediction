# ============================================================
# DIABETES PREDICTION — STREAMLIT APP
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl",  "rb") as f: model   = pickle.load(f)
    with open("scaler.pkl", "rb") as f: scaler  = pickle.load(f)
    with open("features.pkl","rb") as f: features = pickle.load(f)
    return model, scaler, features

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df[col] = df[col].replace(0, df[col].replace(0, np.nan).median())
    df['BMI_Age']          = df['BMI'] * df['Age']
    df['Glucose_Insulin']  = df['Glucose'] / (df['Insulin'] + 1)
    df['HighGlucose']      = (df['Glucose'] > 140).astype(int)
    return df

try:
    model, scaler, feature_cols = load_model()
    df = load_data()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=60)
st.sidebar.title("🩺 Diabetes Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 EDA Dashboard", "🤖 Predict", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Prathmesh Birajdar**")
st.sidebar.markdown("B.Tech AI & Data Science, PCU")
st.sidebar.markdown("[GitHub](https://github.com/PrathmeshBirajdar74) | [LinkedIn](https://linkedin.com/in/prathmesh-birajdar-72b4823a9)")

# ── HOME PAGE ─────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction System")
    st.markdown("#### Using Machine Learning to predict diabetes risk from clinical data")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Records", f"{len(df):,}")
    with col2:
        st.metric("🧪 Features Used", "11")
    with col3:
        diabetic_pct = round(df['Outcome'].mean() * 100, 1)
        st.metric("🔴 Diabetic Cases", f"{diabetic_pct}%")
    with col4:
        st.metric("🤖 Model", "Random Forest")

    st.markdown("---")
    st.markdown("### What this app does")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info("**📊 EDA Dashboard**\n\nExplore the dataset with interactive charts, distributions, and correlation analysis.")
    with col_b:
        st.success("**🤖 Predict**\n\nEnter a patient's clinical values and get an instant diabetes risk prediction with probability.")
    with col_c:
        st.warning("**📈 Model Info**\n\nSee feature importances, confusion matrix, and model performance metrics.")

    st.markdown("---")
    st.markdown("### Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

# ── EDA DASHBOARD ─────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Feature Importance", "Class Balance"])

    with tab1:
        st.subheader("Feature Distributions by Outcome")
        try:
            st.image("eda_distributions.png", use_column_width=True)
        except:
            # Generate on the fly if image not found
            features_plot = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'Pregnancies']
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            for idx, feat in enumerate(features_plot):
                ax = axes[idx // 3][idx % 3]
                df[df['Outcome']==0][feat].hist(ax=ax, alpha=0.6, color='steelblue', label='Non-Diabetic', bins=20)
                df[df['Outcome']==1][feat].hist(ax=ax, alpha=0.6, color='tomato',    label='Diabetic',     bins=20)
                ax.set_title(feat, fontweight='bold')
                ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

    with tab2:
        st.subheader("Feature Correlation Heatmap")
        try:
            st.image("eda_correlation.png", width=700)
        except:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
            st.pyplot(fig)
        st.info("💡 **Glucose** has the highest positive correlation with diabetes outcome (0.49), followed by **BMI** and **Age**.")

    with tab3:
        st.subheader("Feature Importances")
        try:
            st.image("feature_importance.png", width='stretch')
        except:
            st.warning("Run eda_and_model.py first to generate this chart.")

    with tab4:
        st.subheader("Class Balance")
        try:
            st.image("eda_class_distribution.png", width='stretch')
        except:
            fig, ax = plt.subplots(figsize=(5, 5))
            counts = df['Outcome'].value_counts()
            ax.pie(counts, labels=['Non-Diabetic', 'Diabetic'],
                   autopct='%1.1f%%', colors=['steelblue', 'tomato'])
            st.pyplot(fig)

    # Stats table
    st.markdown("---")
    st.subheader("Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ── PREDICTION PAGE ───────────────────────────────────────
elif page == "🤖 Predict":
    st.title("🤖 Diabetes Risk Prediction")
    st.markdown("Enter the patient's clinical values below:")
    st.markdown("---")

    if not model_loaded:
        st.error("⚠️ Model not found. Please run `python eda_and_model.py` first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.slider("Pregnancies", 0, 17, 3, help="Number of times pregnant")
            glucose     = st.slider("Glucose (mg/dL)", 44, 199, 120, help="Plasma glucose concentration")
            bp          = st.slider("Blood Pressure (mm Hg)", 24, 122, 70, help="Diastolic blood pressure")
            skin        = st.slider("Skin Thickness (mm)", 7, 99, 23, help="Triceps skin fold thickness")
            insulin     = st.slider("Insulin (μU/mL)", 14, 846, 80, help="2-Hour serum insulin")

        with col2:
            bmi         = st.slider("BMI", 18.0, 67.0, 32.0, step=0.1, help="Body mass index")
            dpf         = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47, step=0.001, help="Genetic risk score")
            age         = st.slider("Age (years)", 21, 81, 33, help="Patient age")

        st.markdown("---")

        if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
            # Engineer features
            bmi_age         = bmi * age
            glucose_insulin = glucose / (insulin + 1)
            high_glucose    = 1 if glucose > 140 else 0

            input_data = np.array([[pregnancies, glucose, bp, skin, insulin,
                                    bmi, dpf, age, bmi_age, glucose_insulin, high_glucose]])
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            diabetic_prob = round(probability[1] * 100, 1)

            st.markdown("---")
            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                if prediction == 1:
                    st.error(f"### 🔴 HIGH RISK\nDiabetic")
                else:
                    st.success(f"### 🟢 LOW RISK\nNon-Diabetic")
            with col_r2:
                st.metric("Diabetes Probability", f"{diabetic_prob}%")
            with col_r3:
                st.metric("Confidence", f"{max(probability)*100:.1f}%")

            # Risk bar
            st.markdown("#### Risk Level")
            st.progress(int(diabetic_prob))

            # Key risk factors
            st.markdown("#### Key values for this patient")
            risk_df = pd.DataFrame({
                "Feature": ["Glucose", "BMI", "Age", "Insulin", "Blood Pressure"],
                "Patient Value": [glucose, bmi, age, insulin, bp],
                "Dataset Average": [
                    round(df['Glucose'].mean(), 1),
                    round(df['BMI'].mean(), 1),
                    round(df['Age'].mean(), 1),
                    round(df['Insulin'].mean(), 1),
                    round(df['BloodPressure'].mean(), 1)
                ]
            })
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

            st.warning("⚠️ This is a machine learning prediction for educational purposes only. Always consult a qualified doctor for medical diagnosis.")

# ── ABOUT PAGE ────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")

    st.markdown("""
    ### 🩺 Diabetes Prediction System

    This project uses the **Pima Indians Diabetes Dataset** (UCI ML Repository) to predict
    whether a patient is likely to have diabetes based on clinical measurements.

    ---

    ### 🔧 Tech Stack
    | Component | Technology |
    |-----------|-----------|
    | Language | Python 3.x |
    | ML Models | Scikit-learn (Logistic Regression, Random Forest) |
    | Data Processing | Pandas, NumPy |
    | Visualization | Matplotlib, Seaborn |
    | Web App | Streamlit |
    | Version Control | Git / GitHub |

    ---

    ### 📊 Dataset
    - **Source**: Pima Indians Diabetes Database (UCI ML Repository / Kaggle)
    - **Records**: 768 patients
    - **Features**: 8 clinical measurements + 3 engineered features
    - **Target**: Binary — Diabetic (1) or Non-Diabetic (0)
    - ⚠️ **Limitation**: This dataset only contains female patients, so predictions may not generalize well to male patients.

    ---

    ### 🤖 Model Performance
    - **Algorithm**: Random Forest Classifier (100 estimators)
    - **Accuracy**: 73.38% on test set
    - **ROC-AUC Score**: 0.8122 
    - **Evaluation**: 5-fold cross-validation, ROC-AUC score, confusion matrix

    ---

    ### 👨‍💻 Built By
    **Prathmesh Birajdar** — B.Tech AI & Data Science, PCU Pune (3rd Year)

    [GitHub](https://github.com/PrathmeshBirajdar74) | [LinkedIn](https://linkedin.com/in/prathmesh-birajdar-72b4823a9)
    """)
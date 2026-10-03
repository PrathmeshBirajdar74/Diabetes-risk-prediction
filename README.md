# 🩺 Diabetes Risk Prediction System

A machine learning web application that predicts diabetes risk from routine clinical measurements, built with Python and deployed with Streamlit.

**🔗 Live App:** [prathmesh-diabetes-predictor.streamlit.app](https://prathmesh-diabetes-predictor.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project builds a binary classification model to predict whether a patient is likely to have diabetes, based on 8 routine clinical measurements. It benchmarks **Random Forest** against **Logistic Regression** on the Pima Indians Diabetes dataset, with the trained model deployed as an interactive web app featuring real-time predictions and an exploratory data analysis dashboard.

## ✨ Features

- **📊 EDA Dashboard** — Interactive charts covering feature distributions, correlation heatmap, feature importance, and class balance
- **🤖 Real-Time Prediction** — Enter a patient's clinical values and get an instant risk classification with probability score
- **📈 Model Transparency** — Confusion matrix, ROC-AUC score, and feature importance all visible in-app
- **⚠️ Honest Limitations** — The app explicitly states dataset constraints rather than overselling accuracy

## 🎯 Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Test Accuracy | 69.48% | **73.38%** |
| ROC-AUC Score | 0.8044 | **0.8122** |
| 5-Fold CV Accuracy | 77.37% | 74.92% |
| Precision (Diabetic) | 0.58 | **0.65** |
| Recall (Diabetic) | 0.48 | **0.52** |
| F1-Score (Diabetic) | 0.53 | **0.58** |

**Random Forest** was selected as the production model based on ROC-AUC and per-class performance.

### Top Predictive Features
1. **Glucose** — by far the strongest signal (importance ≈ 0.19)
2. **BMI × Age** *(engineered)* — ranked above both raw BMI and raw Age individually
3. **BMI**
4. **Diabetes Pedigree Function**

## 📂 Dataset

[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (UCI ML Repository) — 768 records, 8 clinical features, binary outcome.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Genetic predisposition score |
| Age | Age in years |

> ⚠️ **Limitation:** This dataset only contains female patients, so predictions may not generalize well to male patients.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest)
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git, GitHub

## 🔬 Methodology

1. **Data Cleaning** — Replaced physiologically invalid zero-values (Glucose, BloodPressure, SkinThickness, Insulin, BMI) with column medians
2. **Feature Engineering** — Created 3 derived features:
   - `BMI_Age` = BMI × Age
   - `Glucose_Insulin` = Glucose ÷ Insulin (insulin resistance proxy)
   - `HighGlucose` = binary flag for Glucose > 140 mg/dL
3. **Train/Test Split** — 80/20 stratified split (614 train / 154 test records)
4. **Scaling** — StandardScaler fit on training data only
5. **Model Training** — Logistic Regression and Random Forest, evaluated with 5-fold cross-validation
6. **Evaluation** — Accuracy, ROC-AUC, precision, recall, F1-score, confusion matrix

## 📁 Project Structure

```
diabetes-risk-prediction/
├── app.py                        # Streamlit web application
├── eda_and_model.py               # Data preprocessing, EDA, training, evaluation
├── diabetes.csv                   # Dataset
├── model.pkl                      # Trained Random Forest model
├── scaler.pkl                     # Fitted StandardScaler
├── features.pkl                   # Ordered feature list
├── requirements.txt               # Python dependencies
├── confusion_matrix.png
├── feature_importance.png
├── eda_correlation.png
├── eda_distributions.png
├── eda_class_distribution.png
└── README.md
```

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/PrathmeshBirajdar74/Diabetes-risk-prediction.git
cd Diabetes-risk-prediction

# Install dependencies
pip install -r requirements.txt

# Train the model (generates model.pkl, scaler.pkl, and chart images)
python eda_and_model.py

# Launch the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

## ⚠️ Known Limitations

- **Population constraint** — trained exclusively on female patients aged 21+ of Pima Indian heritage
- **Moderate recall (52%)** on the diabetic class — roughly half of true diabetic cases are missed at the default threshold
- **Small dataset** (768 records) — evaluation metrics carry wide confidence intervals
- **No hyperparameter tuning** — model uses default Scikit-learn parameters

## 🔭 Future Improvements

- [ ] Address class imbalance via SMOTE, class weighting, or threshold tuning to improve recall
- [ ] Hyperparameter optimization via grid/randomized search
- [ ] Benchmark XGBoost and LightGBM
- [ ] Add SHAP values for per-prediction explainability
- [ ] Validate on a more demographically diverse dataset

## 📜 Disclaimer

This application is built for **educational purposes only** and does not constitute medical advice. Always consult a qualified healthcare professional for medical diagnosis and decisions.

## 👤 Author

**Prathmesh Birajdar**
B.Tech, Computer Science (AI & Data Science) — Pimpri Chinchwad University, Pune

- GitHub: [@PrathmeshBirajdar74](https://github.com/PrathmeshBirajdar74)
- LinkedIn: [Prathmesh Birajdar](https://linkedin.com/in/prathmesh-birajdar-72b4823a9)

---

⭐ If you found this project useful, consider giving it a star!

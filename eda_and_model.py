# ============================================================
# DIABETES PREDICTION PROJECT
# Step 1: EDA + Model Training + Save Model
# Run this file FIRST before app.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   DIABETES PREDICTION — EDA & MODEL TRAINING")
print("=" * 55)

# ── 1. LOAD DATA ──────────────────────────────────────────
df = pd.read_csv("diabetes.csv")
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ── 2. BASIC INFO ─────────────────────────────────────────
print("\n\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
print(df.describe().round(2))

print(f"\n🎯 Target Distribution:")
print(df['Outcome'].value_counts())
print(f"   → {(df['Outcome']==0).sum()} Non-Diabetic  |  {(df['Outcome']==1).sum()} Diabetic")

# ── 3. HANDLE MISSING / ZERO VALUES ───────────────────────
# In this dataset, 0s in medical columns = missing values
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print(f"\n🔧 Replacing 0s with median in: {zero_cols}")
for col in zero_cols:
    median_val = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, median_val)
    print(f"   {col}: median = {median_val:.1f}")

# ── 4. EDA VISUALIZATIONS ─────────────────────────────────
print("\n📉 Generating EDA plots...")
sns.set_style("whitegrid")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Diabetes Dataset — Exploratory Data Analysis", fontsize=16, fontweight='bold', y=1.01)

features = ['Pregnancies', 'Glucose', 'BloodPressure',
            'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age']

for idx, feature in enumerate(features):
    ax = axes[idx // 3][idx % 3]
    df[df['Outcome'] == 0][feature].hist(ax=ax, alpha=0.6, color='steelblue', label='Non-Diabetic', bins=20)
    df[df['Outcome'] == 1][feature].hist(ax=ax, alpha=0.6, color='tomato', label='Diabetic', bins=20)
    ax.set_title(feature, fontweight='bold')
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

# Hide the last empty subplot
axes[2][2].set_visible(False)
plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved: eda_distributions.png")

# Correlation heatmap
fig2, ax2 = plt.subplots(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax2, square=True, linewidths=0.5)
ax2.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved: eda_correlation.png")

# Outcome pie chart
fig3, ax3 = plt.subplots(figsize=(6, 6))
counts = df['Outcome'].value_counts()
ax3.pie(counts, labels=['Non-Diabetic', 'Diabetic'],
        autopct='%1.1f%%', colors=['steelblue', 'tomato'],
        startangle=90, textprops={'fontsize': 13})
ax3.set_title("Dataset Class Distribution", fontsize=14, fontweight='bold')
plt.savefig("eda_class_distribution.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved: eda_class_distribution.png")

# ── 5. FEATURE ENGINEERING ────────────────────────────────
print("\n⚙️  Feature engineering...")
df['BMI_Age'] = df['BMI'] * df['Age']
df['Glucose_Insulin'] = df['Glucose'] / (df['Insulin'] + 1)
df['HighGlucose'] = (df['Glucose'] > 140).astype(int)

# ── 6. PREPARE DATA ───────────────────────────────────────
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                'BMI_Age', 'Glucose_Insulin', 'HighGlucose']

X = df[feature_cols]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   Train set: {X_train.shape[0]} rows")
print(f"   Test set : {X_test.shape[0]} rows")

# ── 7. TRAIN & COMPARE MODELS ─────────────────────────────
print("\n🤖 Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc     = roc_auc_score(y_test, y_proba)
    cv      = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy').mean()

    results[name] = {"model": model, "acc": acc, "roc": roc, "cv": cv, "pred": y_pred}
    print(f"\n   {name}:")
    print(f"     Accuracy        : {acc*100:.2f}%")
    print(f"     ROC-AUC Score   : {roc:.4f}")
    print(f"     CV Accuracy (5k): {cv*100:.2f}%")
    print(f"\n   Classification Report:\n{classification_report(y_test, y_pred, target_names=['Non-Diabetic','Diabetic'])}")

# ── 8. PICK BEST MODEL ────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['roc'])
best      = results[best_name]
print(f"\n🏆 Best model: {best_name}  (ROC-AUC: {best['roc']:.4f})")

# Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig4, ax4 = plt.subplots(figsize=(9, 5))
importances.plot(kind='bar', color='steelblue', ax=ax4)
ax4.set_title("Feature Importances — Random Forest", fontsize=13, fontweight='bold')
ax4.set_ylabel("Importance Score")
ax4.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved: feature_importance.png")

# Confusion matrix
fig5, ax5 = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
ax5.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight='bold')
ax5.set_ylabel("Actual")
ax5.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
print("   ✅ Saved: confusion_matrix.png")

# ── 9. SAVE MODEL & SCALER ────────────────────────────────
with open("model.pkl",  "wb") as f: pickle.dump(best['model'], f)
with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)
with open("features.pkl", "wb") as f: pickle.dump(feature_cols, f)

print(f"\n✅ Saved: model.pkl  |  scaler.pkl  |  features.pkl")
print(f"\n{'='*55}")
print(f"  ✅ EDA & Training complete!")
print(f"  Best Model : {best_name}")
print(f"  Accuracy   : {best['acc']*100:.2f}%")
print(f"  ROC-AUC    : {best['roc']:.4f}")
print(f"  Now run:   streamlit run app.py")
print(f"{'='*55}\n")
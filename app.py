import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# -------------------------------
# Helper Functions
# -------------------------------
def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    roc_fig, ax = plt.subplots()
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig)

        # ROC Curve
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.2f})")

    ax.plot([0,1], [0,1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(roc_fig)

    return results

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Disease Prediction Toolkit", layout="wide")
st.title("🩺 Disease Prediction Toolkit")

st.write("Upload a healthcare dataset, select the target column, train ML models, and view evaluation metrics & visualizations.")

uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔎 Data Preview")
    st.write(df.head())

    target_col = st.selectbox("🎯 Select the target column", df.columns)

    if st.button("🚀 Train & Evaluate Models"):
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df, target_col)
        models = train_models(X_train, y_train)
        results = evaluate_models(models, X_test, y_test)

        st.subheader("📊 Model Performance Metrics")
        st.json(results)

        # Save models & scaler
        joblib.dump(scaler, "scaler.joblib")
        for name, model in models.items():
            joblib.dump(model, f"{name.replace(' ', '_').lower()}.joblib")
        st.success("Models and scaler saved locally as .joblib files.")

# app.py
import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "heart_model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_CSV = "heart_disease_health_indicators.csv"

# ————————————————————————————
# Feature order (MUST match training)
# ————————————————————————————
FEATURE_ORDER = [
    "HighBP", "HighChol", "BMI", "Smoker", "Diabetes",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "MentHlth", "PhysHlth", "Sex", "Age", "Education", "Income"
]

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")


@st.cache_resource
def load_saved_model():
    """Try load model & scaler from disk. Return (model, scaler) or (None, None)."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler
    except Exception as e:
        # Corrupt model file, ignore and retrain
        st.warning(f"Saved model load failed: {e}. Will retrain from CSV if available.")
    return None, None


def train_and_save_model(csv_path):
    """Train model from CSV and save model+scaler. Returns (model, scaler)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found at '{csv_path}'")

    df = pd.read_csv(csv_path)

    # Identify target column — common names: 'HeartDisease' or 'HeartDiseaseor something.
    # We'll try a few possibilities then ask user if not found.
    possible_targets = ["HeartDisease", "HeartDiseaseor", "HeartDiseaseBinary", "HeartDiseaseorHere", "HeartDiseaseor"]
    target_col = None
    for c in ["HeartDisease", "HeartDiseaseor", "Heart_Disease", "HeartDisease_binary", "HeartDiseaseBinary", "HeartDiseaseorHere"]:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        # fallback: assume last column is target
        target_col = df.columns[-1]
        st.warning(f"Could not find a standard target column — using last column '{target_col}' as target. "
                   "If this is wrong, edit the CSV or adjust the code.")

    # Ensure feature columns exist
    missing = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        raise KeyError(f"Training CSV is missing these required columns: {missing}")

    X = df[FEATURE_ORDER].values
    y = df[target_col].values

    # basic train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Use KNeighborsClassifier (as in your notebook)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_s, y_train)

    # quick accuracy check
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    # save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    st.info(f"Trained KNeighborsClassifier and saved model & scaler (test accuracy ≈ {acc:.3f})")

    return model, scaler


@st.cache_resource
def get_model_and_scaler():
    model, scaler = load_saved_model()
    if model is not None and scaler is not None:
        return model, scaler

    # else train
    model, scaler = train_and_save_model(DATA_CSV)
    return model, scaler


# ————————————————————————————
# Build UI
# ————————————————————————————
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("Predict likelihood of heart disease from health indicators. Fill inputs and press Predict.")

# Try to load or train model
try:
    model, scaler = get_model_and_scaler()
except FileNotFoundError as e:
    st.error(f"Model not found and cannot train: {e}")
    st.stop()
except KeyError as e:
    st.error(f"Training data columns problem: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading/training model: {e}")
    st.stop()

# Input widgets (same order as FEATURE_ORDER)
cols = st.columns(3)
HighBP = cols[0].selectbox("HighBP (0=no, 1=yes)", [0, 1], index=0)
HighChol = cols[0].selectbox("HighChol (0=no, 1=yes)", [0, 1], index=0)
BMI = cols[0].number_input("BMI", min_value=5.0, max_value=60.0, value=25.0, step=0.1)

Smoker = cols[1].selectbox("Smoker (0=no, 1=yes)", [0, 1], index=0)
Diabetes = cols[1].selectbox("Diabetes (0=no, 1=yes)", [0, 1], index=0)
PhysActivity = cols[1].selectbox("PhysActivity (0=no, 1=yes)", [0, 1], index=1)

Fruits = cols[2].selectbox("Fruits (0=no, 1=yes)", [0, 1], index=1)
Veggies = cols[2].selectbox("Veggies (0=no, 1=yes)", [0, 1], index=1)
HvyAlcoholConsump = cols[2].selectbox("Heavy Alcohol Consumption (0=no,1=yes)", [0, 1], index=0)

MentHlth = st.slider("MentHlth (bad mental health days in last 30 days)", 0, 30, 2)
PhysHlth = st.slider("PhysHlth (bad physical health days in last 30 days)", 0, 30, 1)

Sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1], index=0)
Age = st.number_input("Age (years)", min_value=18, max_value=120, value=45, step=1)
Education = st.slider("Education level (1-6)", 1, 6, 3)
Income = st.slider("Income level (1-8)", 1, 8, 4)

# Create feature array in exact order
features = np.array([[HighBP, HighChol, BMI, Smoker, Diabetes, PhysActivity,
                      Fruits, Veggies, HvyAlcoholConsump, MentHlth, PhysHlth,
                      Sex, Age, Education, Income]], dtype=float)

# Predict
if st.button("🔍 Predict"):
    try:
        features_s = scaler.transform(features)
        pred = model.predict(features_s)[0]
        proba = model.predict_proba(features_s) if hasattr(model, "predict_proba") else None

        if int(pred) == 1:
            st.error("⚠️ Model predicts a risk of heart disease.")
        else:
            st.success("✅ Model predicts you are likely healthy.")

        if proba is not None:
            st.write(f"Model probabilities: {proba[0].round(3)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

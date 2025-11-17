# app.py (FINAL CLEAN VERSION)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Correct model file names
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Feature order must match training
FEATURE_ORDER = [
    "HighBP", "HighChol", "BMI", "Smoker", "Diabetes",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "MentHlth", "PhysHlth", "Sex", "Age", "Education", "Income"
]

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# -------------------------------------------------------
# Load saved model and scaler
# -------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ model.pkl not found! Upload it to your repo.")
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error(f"❌ scaler.pkl not found! Upload it to your repo.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model_and_scaler()

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("Fill in the details and click **Predict**.")

cols = st.columns(3)

HighBP = cols[0].selectbox("HighBP (0=no, 1=yes)", [0, 1])
HighChol = cols[0].selectbox("HighChol (0=no, 1=yes)", [0, 1])
BMI = cols[0].number_input("BMI", min_value=5.0, max_value=60.0, value=25.0, step=0.1)

Smoker = cols[1].selectbox("Smoker (0=no, 1=yes)", [0, 1])
Diabetes = cols[1].selectbox("Diabetes (0=no, 1=yes)", [0, 1])
PhysActivity = cols[1].selectbox("PhysActivity (0=no, 1=yes)", [0, 1])

Fruits = cols[2].selectbox("Fruits (0=no, 1=yes)", [0, 1])
Veggies = cols[2].selectbox("Veggies (0=no, 1=yes)", [0, 1])
HvyAlcoholConsump = cols[2].selectbox("Heavy Alcohol Consumption (0=no,1=yes)", [0, 1])

MentHlth = st.slider("MentHlth (bad mental health days in last 30 days)", 0, 30, 2)
PhysHlth = st.slider("PhysHlth (bad physical health days in last 30 days)", 0, 30, 1)

Sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
Age = st.number_input("Age (years)", min_value=18, max_value=120, value=45)
Education = st.slider("Education level (1-6)", 1, 6, 3)
Income = st.slider("Income level (1-8)", 1, 8, 4)

# Create feature vector
features = np.array([[HighBP, HighChol, BMI, Smoker, Diabetes, PhysActivity,
                      Fruits, Veggies, HvyAlcoholConsump, MentHlth, PhysHlth,
                      Sex, Age, Education, Income]], dtype=float)

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
if st.button("🔍 Predict"):
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        if prediction == 1:
            st.error("⚠️ Model predicts a **high risk** of heart disease.")
        else:
            st.success("✅ Model predicts **low risk** of heart disease.")

        st.write("### Probability")
        st.write(f"Healthy: {proba[0]:.3f} | Risk: {proba[1]:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("""
---
### ✨ Created by **Mansi Pare**
""")

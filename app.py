import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd

# -----------------------------
# Step 1: Page configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction App ❤️",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts your **risk of heart disease** based on health and lifestyle inputs.  
Please fill in the following details carefully.
""")

# -----------------------------
# Step 2: Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "heart_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file 'heart_model.pkl' not found in project folder.")
        st.stop()
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        st.stop()

@st.cache_resource
def load_scaler():
    scaler_path = "scaler.pkl"
    if not os.path.exists(scaler_path):
        st.error("❌ Scaler file 'scaler.pkl' not found in project folder.")
        st.stop()
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"⚠️ Failed to load scaler: {e}")
        st.stop()

model = load_model()
scaler = load_scaler()
st.success("✅ Model and Scaler loaded successfully!")

# -----------------------------
# Step 3: Collect User Inputs
# -----------------------------
st.subheader("🧍 Enter Your Health Details")

col1, col2 = st.columns(2)

with col1:
    HighBP = st.selectbox('High Blood Pressure?', [0, 1])
    HighChol = st.selectbox('High Cholesterol?', [0, 1])
    BMI = st.number_input('BMI value', min_value=10.0, max_value=60.0, step=0.1)
    Smoker = st.selectbox('Smoker?', [0, 1])
    Diabetes = st.selectbox('Diabetes?', [0, 1])
    PhysActivity = st.selectbox('Exercise regularly?', [0, 1])
    Fruits = st.selectbox('Eat fruits daily?', [0, 1])

with col2:
    Veggies = st.selectbox('Eat vegetables daily?', [0, 1])
    HvyAlcoholConsump = st.selectbox('Heavy Alcohol Consumption?', [0, 1])
    MentHlth = st.number_input('Bad mental health days (past 30 days)', 0, 30)
    PhysHlth = st.number_input('Bad physical health days (past 30 days)', 0, 30)
    Sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    Age = st.number_input('Age (in years)', min_value=18, max_value=100, step=1)
    Education = st.slider('Education Level (1–6)', 1, 6)
    Income = st.slider('Income Level (1–8)', 1, 8)

# -----------------------------
# Step 4: Prepare Data
# -----------------------------
features = np.array([[HighBP, HighChol, BMI, Smoker, Diabetes, PhysActivity,
                      Fruits, Veggies, HvyAlcoholConsump, MentHlth, PhysHlth,
                      Sex, Age, Education, Income]])

if st.checkbox("👀 Show Entered Details"):
    st.write(pd.DataFrame(features, columns=[
        'HighBP','HighChol','BMI','Smoker','Diabetes','PhysActivity','Fruits',
        'Veggies','HvyAlcoholConsump','MentHlth','PhysHlth','Sex','Age','Education','Income'
    ]))

# -----------------------------
# Step 5: Prediction
# -----------------------------
if st.button('🔍 Predict'):
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            st.error("⚠️ You may be at **risk of heart disease.** Please consult a doctor.")
        else:
            st.success("✅ You are likely **healthy.** Keep maintaining a good lifestyle!")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -----------------------------
# Step 6: Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by **Mansi ❤️** | Powered by Streamlit & Scikit-learn")

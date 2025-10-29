import streamlit as st
import numpy as np
import joblib
import requests
import io
import os

# Step 1: Download and load model directly from Google Drive
model_url = 'https://drive.google.com/uc?export=download&id=1145pyLGPoikAtEn6sK0kRN8TaOvolyYy'

@st.cache_resource
def load_model():
    response = requests.get(model_url)
    response.raise_for_status()
    model = joblib.load(io.BytesIO(response.content))
    return model

model = load_model()

# Step 2: Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

# Step 3: Streamlit app
st.title("❤️ Heart Disease Prediction App")

# Step 4: Collect user input
HighBP = st.selectbox('Do you have High Blood Pressure?', [0, 1])
HighChol = st.selectbox('Do you have High Cholesterol?', [0, 1])
BMI = st.number_input('Enter BMI value', min_value=10.0, max_value=60.0, step=0.1)
Smoker = st.selectbox('Are you a Smoker?', [0, 1])
Diabetes = st.selectbox('Do you have Diabetes?', [0, 1])
PhysActivity = st.selectbox('Do you exercise regularly?', [0, 1])
Fruits = st.selectbox('Do you eat fruits daily?', [0, 1])
Veggies = st.selectbox('Do you eat vegetables daily?', [0, 1])
HvyAlcoholConsump = st.selectbox('Do you heavily consume alcohol?', [0, 1])
MentHlth = st.number_input('How many bad mental health days (past 30 days)?', 0, 30)
PhysHlth = st.number_input('How many bad physical health days (past 30 days)?', 0, 30)
Sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
Age = st.number_input('Age (in years)', min_value=18, max_value=100, step=1)
Education = st.slider('Education Level (1–6)', 1, 6)
Income = st.slider('Income Level (1–8)', 1, 8)

# Step 5: Prepare data
features = np.array([[HighBP, HighChol, BMI, Smoker, Diabetes, PhysActivity,
                      Fruits, Veggies, HvyAlcoholConsump, MentHlth, PhysHlth,
                      Sex, Age, Education, Income]])

# Step 6: Predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)

# Step 7: Show result
if st.button('Predict'):
    if prediction[0] == 1:
        st.error("⚠️ You may be at risk of heart disease.")
    else:
        st.success("✅ You are likely healthy.")

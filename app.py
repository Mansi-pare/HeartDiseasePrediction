import streamlit as st
import numpy as np
import pickle
import gdown
import os

# Step 1: Download model file from Google Drive
model_url = 'https://drive.google.com/uc?id=12PhO6lWSpPNNlQeD3eqzuu8w6ZFpGL1w'  # Replace with your link
model_path = 'heart_model.pkl'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load model and scaler
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Step 3: Streamlit app
st.title("❤️ Heart Disease Prediction App")

# Step 4: Collect user input in SAME order
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

# Step 5: Arrange inputs in same order as training
features = np.array([[HighBP, HighChol, BMI, Smoker, Diabetes, PhysActivity,
                      Fruits, Veggies, HvyAlcoholConsump, MentHlth, PhysHlth,
                      Sex, Age, Education, Income]])

# Step 6: Scale and Predict
if st.button('Predict'):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.error("⚠️ You may be at risk of heart disease.")
    else:
        st.success("✅ You are likely healthy.")
st.markdown("---")
st.markdown("👩‍💻 Created by **Mansi Pare** — Machine Learning Project on Heart Disease Prediction")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("💓 Heart Disease Prediction Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your heart disease dataset (CSV file)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    st.write("### Data Statistics")
    st.write(df.describe())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.markdown("👩‍⚕️ *Created by Mansi Pare — Machine Learning Project on Heart Disease Prediction*")

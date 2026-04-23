import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# LOAD MODEL
# -----------------------------
with open("model1.pkl", "rb") as f:
    model, scaler, le, columns = pickle.load(f)

# -----------------------------
# TITLE
# -----------------------------
st.title("🌱 Smart Agriculture Prediction App")
st.write("Enter input values to predict output")

# -----------------------------
# USER INPUT
# -----------------------------
user_input = []

for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    user_input.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    try:
        input_data = pd.DataFrame([user_input], columns=columns)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        # Decode label
        result = le.inverse_transform(prediction)

        st.success(f"Prediction: {result[0]}")

    except Exception as e:
        st.error(f"Error: {e}")

# -----------------------------
# SHOW DATASET (LOCAL FILE)
# -----------------------------
st.subheader("Dataset Preview")

if st.button("Show Dataset"):
    try:
        df = pd.read_csv("Advanced_IoT_Dataset.csv")
        st.dataframe(df.head())
    except:
        st.error("Dataset file not found. Please upload it to repo.")

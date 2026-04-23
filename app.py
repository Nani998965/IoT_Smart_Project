import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
try:
    with open("model1.pkl", "rb") as f:
        data = pickle.load(f)

    # Handle different formats
    if isinstance(data, dict):
        model = data.get("model")
        scaler = data.get("scaler")
        le = data.get("label_encoder")
        columns = data.get("columns")

    elif isinstance(data, tuple):
        if len(data) == 4:
            model, scaler, le, columns = data
        elif len(data) == 2:
            model, scaler = data
            le = None
            columns = None
        else:
            model = data[0]
            scaler = None
            le = None
            columns = None
    else:
        model = data
        scaler = None
        le = None
        columns = None

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# DEFAULT COLUMNS (if missing)
# -----------------------------
if columns is None:
    columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

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

        # Apply scaling if available
        if scaler is not None:
            input_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data)

        # Decode if label encoder exists
        if le is not None:
            prediction = le.inverse_transform(prediction)

        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------------
# SHOW DATASET
# -----------------------------
st.subheader("Dataset Preview")

if st.button("Show Dataset"):
    try:
        df = pd.read_csv("Advanced_IoT_Dataset.csv")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Dataset error: {e}")

import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# LOAD MODEL (joblib is safer than pickle for sklearn models)
# -----------------------------
try:
    import joblib
    model, scaler, le, columns = joblib.load("model1.pkl")
except Exception as joblib_err:
    try:
        import pickle
        with open("model1.pkl", "rb") as f:
            model, scaler, le, columns = pickle.load(f)
    except Exception as pickle_err:
        st.error("❌ Failed to load model. Please re-save your model using joblib:")
        st.code("""
import joblib
joblib.dump((model, scaler, le, columns), "model1.pkl")
        """, language="python")
        st.stop()

# -----------------------------
# TITLE
# -----------------------------
st.set_page_config(page_title="Smart Agriculture Prediction", page_icon="🌱")
st.title("🌱 Smart Agriculture Prediction App")
st.write("Enter sensor/input values below to get a crop or condition prediction.")

# -----------------------------
# USER INPUT (AUTO FROM COLUMNS)
# -----------------------------
st.subheader("📥 Input Features")

user_input = []
col1, col2 = st.columns(2)

for i, col in enumerate(columns):
    with col1 if i % 2 == 0 else col2:
        val = st.number_input(f"{col}", value=0.0, key=col)
        user_input.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
st.markdown("---")
if st.button("🔍 Predict", use_container_width=True):
    try:
        input_data = pd.DataFrame([user_input], columns=columns)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = le.inverse_transform(prediction)
        st.success(f"✅ Prediction: **{result[0]}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------------
# OPTIONAL: SHOW DATASET
# -----------------------------
st.markdown("---")
st.subheader("📊 Dataset Preview")

if st.button("Show Dataset", use_container_width=True):
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "wisam1985/advanced-iot-agriculture-2024",
            "Advanced_IoT_Dataset.csv"
        )
        st.dataframe(df.head())
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")

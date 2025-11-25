import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/crop_yield_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run `python training.py` first.")
        return None
    return joblib.load(MODEL_PATH)

def main():
    st.title("🌾 Crop Yield Prediction")

    col1, col2 = st.columns(2)
    with col1:
        rainfall = st.number_input("Rainfall (mm)", 0.0, 2000.0, 800.0)
        fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 500.0, 150.0)
    with col2:
        temp = st.number_input("Avg Temp (°C)", -10.0, 50.0, 26.0)
        soil_ph = st.number_input("Soil pH", 3.0, 10.0, 6.5)

    model = load_model()

    if st.button("Predict Yield"):
        if model is None:
            st.stop()
        input_df = pd.DataFrame([{
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "fertilizer_kg_per_ha": fertilizer,
            "soil_ph": soil_ph
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"🌱 Predicted Yield: **{prediction:.2f} tons/ha**")

if __name__ == "__main__":
    main()

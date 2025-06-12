import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('polution_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Pollution Level Predictor")

month = st.number_input("Enter Month (1-12)", min_value=1, max_value=12, step=1)
dayofweek = st.number_input("Enter Day of Week (0=Monday,...6=Sunday)", min_value=0, max_value=6, step=1)

if st.button("Predict"):
    input_features = np.array([[month, dayofweek]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Pollution Level: {prediction[0]:.2f}")

import streamlit as st
import joblib

# Load models and encoders
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")  # Assuming this encodes gender

# Streamlit UI
st.title("Hepatitis C Prediction")

# Input numeric values
data = [st.number_input(col) for col in ['ALT', 'AST', 'GGT', 'PROT']]

# Add gender input
gender = st.selectbox("Gender", ["m", "f"])

# On button click, process and predict
if st.button("Predict"):
    # Encode gender
    encoded_gender = encoder.transform([gender])[0]  # Encode as numerical

    # Append encoded gender to data
    input_data = [encoded_gender] + data

    # Scale and predict
    scaled_data = scaler.transform([input_data])
    result = model.predict(scaled_data)

    # Output result
    st.write("Prediction:", "Hepatitis C" if result[0] == 1 else "Healthy")

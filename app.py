import streamlit as st
import joblib

model = joblib.load("/content/best_model.pkl")
scaler = joblib.load("/content/scaler.pkl")

st.title("Hepatitis C Prediction")

data = [st.number_input(col) for col in ['ALT', 'AST', 'BIL', 'CHE', 'GGT']]
if st.button("Predict"):
    result = model.predict(scaler.transform([data]))
    st.write("Prediction:", "Hepatitis C" if result[0] == 1 else "Healthy")
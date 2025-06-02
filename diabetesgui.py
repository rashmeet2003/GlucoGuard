import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('diabetes_model.pkl')

# App title
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter your health metrics below:")

# Input form
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0)
    insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"🔴 Positive for Diabetes with {probability*100:.2f}% confidence")
    else:
        st.success(f"🟢 Negative for Diabetes with {(1 - probability)*100:.2f}% confidence")

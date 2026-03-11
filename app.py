import pandas as pd
import streamlit as st
import joblib

model = joblib.load('Logistic-Regression.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction App")
st.markdown("Add the details below to predict the chances of heart disease")

age = st.slider("Age", 30, 65, 85)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_blood_pressure = st.slider("Resting Blood Pressure", 90, 200, 150)
cholesterol = st.slider("Cholesterol", 100, 400, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
resting_ecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 200, 100)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["True", "False"])
oldpeak = st.slider("ST depression induced by exercise relative to rest", 0.0, 6.2, 1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict"):
    input_data = {
        'age': age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_ecg": resting_ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_angina": exercise_induced_angina,
        "oldpeak": oldpeak,
        "slope": slope,
    }

    input_df = pd.DataFrame([input_data])
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]

    input_df_scaled = scaler.transform(input_df)

    prediction = model.predict(input_df_scaled)[0]

    if prediction == 1:
        st.error("The model predicts that you have a high chance of heart disease. Please consult a doctor for further evaluation.")
    else:
        st.success("The model predicts that you have a low chance of heart disease. However, this is not a definitive diagnosis. Please consult a doctor for a comprehensive evaluation.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="centered")

st.title("❤️ Heart Disease Risk Prediction")
st.write("Machine Learning-based risk assessment using clinical data (UCI Heart Disease Dataset)")

# --- Load saved model, scaler, and column order ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("heart_disease_scaler.pkl")
    columns = joblib.load("heart_disease_columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_artifacts()

st.subheader("Enter Patient Clinical Values")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type", options=[
        ("Typical Angina", 1), ("Atypical Angina", 2),
        ("Non-anginal Pain", 3), ("Asymptomatic", 4)], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Resting ECG", options=[
        ("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2)], format_func=lambda x: x[0])[1]

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", options=[
        ("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)], format_func=lambda x: x[0])[1]
    ca = st.selectbox("Major Vessels Blocked (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[
        ("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)], format_func=lambda x: x[0])[1]

if st.button("Predict Risk"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                 thalach, exang, oldpeak, slope, ca, thal]], columns=columns)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if probability < 0.33:
        risk = "Low"
        color = "green"
    elif probability < 0.66:
        risk = "Medium"
        color = "orange"
    else:
        risk = "High"
        color = "red"

    st.subheader("Result")
    st.markdown(f"### Risk Level: :{color}[{risk}]")
    st.write(f"Predicted Probability of Heart Disease: **{probability:.1%}**")

    # SHAP explanation
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    if isinstance(shap_values, list):
        patient_shap = shap_values[1][0]
    else:
        patient_shap = shap_values[0, :, 1]

    explain_df = pd.DataFrame({
        "Feature": columns,
        "Contribution": patient_shap
    }).sort_values(by="Contribution", key=abs, ascending=False).head(5)

    st.subheader("Why this result? (Top contributing factors)")
    for _, row in explain_df.iterrows():
        direction = "increases" if row["Contribution"] > 0 else "decreases"
        st.write(f"- **{row['Feature']}** {direction} risk")

st.markdown("---")
st.caption("Model: Random Forest | Dataset: UCI Heart Disease (Cleveland) | For educational/demo purposes only.")
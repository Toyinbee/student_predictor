import streamlit as st
import shap
import numpy as np
import pandas as pd
import joblib

# Load all saved models and encoders
best_model_mental = joblib.load("mental_model.pkl")
best_model_cgpa = joblib.load("cgpa_model.pkl")
mental_encoder = joblib.load("mental_encoder .pkl")
cgpa_encoder = joblib.load("cgpa_encoder .pkl")
label_encoder_mental = joblib.load("label_encoder_mental.pkl")
label_encoder_cgpa = joblib.load("label_encoder_cgpa.pkl")

mental_input_features = ['choose_your_gender', 'age', 'what_is_your_course?',
                         'your_current_year_of_study', 'marital_status',
                         'did_you_seek_any_specialist_for_a_treatment?']

cgpa_input_features = mental_input_features + ['has_issue']

# Streamlit app
st.set_page_config(page_title="Mental Health & CGPA Predictor", layout="centered")
st.title("🎓 Student Wellbeing & CGPA Predictor")

with st.form("user_form"):
    name = st.text_input("👤 Name")
    gender = st.selectbox("⚧ Gender", ["Male", "Female"])
    age = st.number_input("🎂 Age", min_value=15, max_value=50, step=1)
    course = st.text_input("📚 Course of Study")
    year = st.selectbox("📅 Year of Study", ["year 1", "year 2", "year 3", "year 4", "year 5"])
    marital = st.selectbox("💍 Marital Status", ["Single", "Married"])
    treatment = st.selectbox("🩺 Sought Treatment?", ["Yes", "No"])
    actual_cgpa = st.number_input("📊 What is your actual CGPA?", min_value=0.0, max_value=5.0, step=0.01)

    depression = st.selectbox("😔 Do you experience depression?", ["Yes", "No"])
    anxiety = st.selectbox("😟 Do you experience anxiety?", ["Yes", "No"])
    panic = st.selectbox("😨 Do you experience panic attacks?", ["Yes", "No"])

    submit = st.form_submit_button("Predict Now")

if submit:
    has_issue_val = 'Has Issue' if "yes" in [depression.lower(), anxiety.lower(), panic.lower()] else 'No Issue'

    user_input = {
        "choose_your_gender": gender,
        "age": age,
        "what_is_your_course?": course,
        "your_current_year_of_study": year,
        "marital_status": marital,
        "did_you_seek_any_specialist_for_a_treatment?": treatment,
        "actual_cgpa": actual_cgpa,
        "name": name,
        "depression": depression,
        "anxiety": anxiety,
        "panic": panic,
        "has_issue": has_issue_val
    }

    df_mental = pd.DataFrame([{k: user_input[k] for k in mental_input_features}])
    df_mental_encoded = pd.DataFrame(mental_encoder.transform(df_mental), columns=mental_input_features)
    mental_pred = best_model_mental.predict(df_mental_encoded)
    mental_label = label_encoder_mental.inverse_transform(mental_pred)[0]

    df_cgpa = pd.DataFrame([{k: user_input[k] for k in cgpa_input_features}])
    df_cgpa_encoded = pd.DataFrame(cgpa_encoder.transform(df_cgpa), columns=cgpa_input_features)
    cgpa_pred = best_model_cgpa.predict(df_cgpa_encoded)
    cgpa_label = label_encoder_cgpa.inverse_transform(cgpa_pred)[0]

    st.markdown("### 🧠 Mental Health Prediction:")
    st.success(f"{mental_label}")
    if mental_label == "Healthy":
        st.info("✅ You seem to be doing okay mentally. Keep it up! 😊")
    else:
        st.warning("⚠️ It looks like you're going through a tough time. You're not alone 💛")

    st.markdown("### 🎓 CGPA Prediction:")
    st.success(f"Predicted CGPA Class: {cgpa_label}")
    st.info(f"Self-reported CGPA: {actual_cgpa}")

    # SHAP Explainability
    explainer_mental = shap.Explainer(best_model_mental, df_mental_encoded)
    shap_values_mental = explainer_mental(df_mental_encoded)
    top_mental = shap_values_mental.values[0]

    st.markdown("### 📌 Why this Mental Health prediction was made:")
    for i in np.argsort(np.abs(top_mental))[::-1][:3]:
        feat = mental_input_features[i]
        val = df_mental.iloc[0][feat]
        direction = "helped reduce" if top_mental[i] < 0 else "may have contributed to"
        st.write(f"- '{val}' for '{feat}' {direction} mental health issues.")

    explainer_cgpa = shap.Explainer(best_model_cgpa, df_cgpa_encoded)
    shap_values_cgpa = explainer_cgpa(df_cgpa_encoded)
    predicted_class_idx = cgpa_pred[0]
    shap_values_for_class = shap_values_cgpa.values[0][:, predicted_class_idx]

    st.markdown("### 📘 Why this CGPA prediction was made:")
    for i in np.argsort(np.abs(shap_values_for_class))[::-1][:3]:
        feat = cgpa_input_features[i]
        val = df_cgpa.iloc[0][feat]
        direction = "helped improve" if shap_values_for_class[i] > 0 else "may have reduced"
        st.write(f"- '{val}' for '{feat}' {direction} your CGPA class.")

    st.markdown("### 📝 Your Mental Health Responses:")
    st.write(f"- 😔 Depression: {depression}")
    st.write(f"- 😟 Anxiety: {anxiety}")
    st.write(f"- 😨 Panic Attacks: {panic}")

    st.markdown("### 💡 Suggestions & Support:")
    if mental_label == "Healthy":
        st.info("👍 Keep taking care of yourself. Stick to good habits like sleep, relaxation, and social connections.")
    else:
        st.info("🧘 Consider talking to a counselor or trusted adult. You're not alone. There are apps, groups, and people ready to help.")

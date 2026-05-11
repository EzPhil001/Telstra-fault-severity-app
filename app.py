import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import uuid

# =========================
# PAGE TITLE
# =========================

st.title("Telstra Fault Severity Prediction with XAI")
st.write("Predicting network fault severity using Machine Learning + Explainable AI")

# =========================
# LOAD MODEL + SCALER
# =========================

model = joblib.load("telstra_xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# USER INPUTS
# =========================

log_feature = st.number_input("Log Feature")
event_type = st.number_input("Event Type")
log_volume = st.number_input("Log Volume")
severity_type = st.number_input("Severity Type")
resource_type = st.number_input("Resource Type")

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame({
    'log_feature': [log_feature],
    'event_type': [event_type],
    'log_volume': [log_volume],
    'severity_type': [severity_type],
    'resource_type': [resource_type]
})

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Fault Severity"):

    # =========================
    # APPLY SCALER (NEW FIX)
    # =========================

    input_scaled = scaler.transform(input_data.values)

    # =========================
    # MODEL PREDICTION
    # =========================

    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction")
    st.success(f"Predicted Fault Severity: {prediction}")

    # =========================
    # CONDITIONAL LOGIC (NEW FEATURE)
    # =========================

    if prediction in [1, 2]:

        st.warning("High severity fault detected — additional details required")

        location = st.text_input("Enter Location")

        if location:

            unique_id = str(uuid.uuid4())

            st.subheader("Fault Tracking Info")
            st.write("Location:", location)
            st.write("Generated Unique ID:", unique_id)

    # =========================
    # SHAP EXPLANATION
    # =========================

    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        input_data,
        show=False
    )

    st.pyplot(fig)

    # =========================
    # LIME EXPLANATION
    # =========================

    st.subheader("LIME Explanation")

    training_data = np.array([
        [0,0,0,0,0],
        [1,1,1,1,1]
    ])

    explainer_lime = LimeTabularExplainer(
        training_data=training_data,
        feature_names=input_data.columns.tolist(),
        class_names=['0','1','2'],
        mode='classification'
    )

    exp = explainer_lime.explain_instance(
        input_scaled[0],
        model.predict_proba,
        num_features=5
    )

    lime_df = pd.DataFrame(
        exp.as_list(),
        columns=['Feature', 'Contribution']
    )

    st.dataframe(lime_df)
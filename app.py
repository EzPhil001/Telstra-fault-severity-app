import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import uuid
from lime.lime_tabular import LimeTabularExplainer

# =========================
# TITLE
# =========================
st.title("Telstra Fault Severity Prediction with XAI")

# =========================
# LOAD MODEL PIPELINE
# =========================
model = joblib.load("xgb_pipeline.pkl")

# =========================
# INPUTS
# =========================
log_feature = st.number_input("Log Feature")
event_type = st.number_input("Event Type")
log_volume = st.number_input("Log Volume")
severity_type = st.number_input("Severity Type")
resource_type = st.number_input("Resource Type")

input_data = pd.DataFrame({
    "log_feature": [log_feature],
    "event_type": [event_type],
    "log_volume": [log_volume],
    "severity_type": [severity_type],
    "resource_type": [resource_type]
})

# =========================
# PREDICTION
# =========================
if st.button("Predict Fault Severity"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction")
    st.success(f"Predicted Fault Severity: {prediction}")

    # =========================
    # HIGH SEVERITY LOGIC
    # =========================
    if prediction in [1, 2]:
        st.warning("High severity fault detected")

        location = st.text_input("Enter Location")

        if location:
            st.write("Location:", location)
            st.write("Ticket ID:", str(uuid.uuid4()))

    # =========================
    # SHAP (CORRECT WAY FOR PIPELINE)
    # =========================
    st.subheader("SHAP Explanation")

    # extract model inside pipeline
    xgb_model = model.named_steps["model"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_data, show=False)
    st.pyplot(fig)

    # =========================
    # LIME (IMPORTANT FIX)
    # =========================
    st.subheader("LIME Explanation")

    dummy_train = np.zeros((10, len(input_data.columns)))

    lime_explainer = LimeTabularExplainer(
        training_data=dummy_train,
        feature_names=input_data.columns.tolist(),
        class_names=["0", "1", "2"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        input_data.iloc[0].values,
        model.predict_proba
    )

    st.dataframe(pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"]))
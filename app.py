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
# LOAD MODEL + FEATURES
# =========================
model = joblib.load("xgb_pipeline.pkl")
feature_columns = joblib.load("features.pkl")

# =========================
# INPUTS
# =========================
log_feature = st.number_input("Log Feature")
event_type = st.number_input("Event Type")
log_volume = st.number_input("Log Volume")
severity_type = st.number_input("Severity Type")
resource_type = st.number_input("Resource Type")

user_input = {
    "log_feature": log_feature,
    "event_type": event_type,
    "log_volume": log_volume,
    "severity_type": severity_type,
    "resource_type": resource_type
}

# IMPORTANT: enforce training feature order
input_data = pd.DataFrame([user_input]).reindex(columns=feature_columns)

## =========================
# PREDICTION
# =========================
if st.button("Predict Fault Severity"):

    prediction = model.predict(input_data)[0]

    st.session_state.prediction = prediction
    st.session_state.prediction_made = True

    st.subheader("Prediction")
    st.success(f"Predicted Fault Severity: {prediction}")

    if prediction in [1, 2]:
        st.warning("High severity fault detected 🚨")
        st.session_state.show_location = True
    else:
        st.session_state.show_location = False
        st.session_state.ticket_id = None

# =========================
# LOCATION + TICKET SYSTEM
# =========================
if st.session_state.show_location:

    location = st.text_input("Enter Location", key="location_input")

    if location:
        if st.session_state.ticket_id is None:
            st.session_state.ticket_id = str(uuid.uuid4())

        st.write("📍 Location:", location)
        st.write("🎫 Ticket ID:", st.session_state.ticket_id)

    # =========================
    # SHAP
    # =========================
    st.subheader("SHAP Explanation")

    xgb_model = model.named_steps["model"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_data, show=False)
    st.pyplot(fig)

    # =========================
    # LIME
    # =========================
    st.subheader("LIME Explanation")

    # Use real training structure approximation (not fake zeros)
    dummy_train = np.zeros((100, len(feature_columns)))

    lime_explainer = LimeTabularExplainer(
        training_data=dummy_train,
        feature_names=feature_columns,
        class_names=["0", "1", "2"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        input_data.iloc[0].values,
        model.predict_proba
    )

    st.dataframe(pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"]))
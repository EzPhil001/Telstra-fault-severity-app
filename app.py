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

xgb_model = model.named_steps["model"]

# =========================
# SESSION STATE INIT
# =========================
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

if "ticket_id" not in st.session_state:
    st.session_state.ticket_id = None

if "show_location" not in st.session_state:
    st.session_state.show_location = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

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

# enforce training feature order
input_data = pd.DataFrame([user_input]).reindex(columns=feature_columns)

# =========================
# PREDICTION
# =========================
if st.button("Predict Fault Severity"):

    prediction = model.predict(input_data)[0]

    st.session_state.prediction = prediction
    st.session_state.prediction_made = True

    st.subheader("Prediction")
    st.success(f"Predicted Fault Severity: {prediction}")

    # HIGH SEVERITY LOGIC
    if prediction in [1, 2]:
        st.warning("High severity fault detected")
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
# SHAP EXPLANATION (FIXED FULL VERSION)
# =========================
if st.session_state.prediction_made:

    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_data)

    # Handle binary vs multiclass safely
    if isinstance(shap_values, list):
        shap_vals = shap_values[st.session_state.prediction]
    else:
        shap_vals = shap_values[0]

    # =========================
    # TABLE VIEW
    # =========================
    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Feature Value": input_data.iloc[0].values,
        "SHAP Impact": shap_vals
    })

    st.write("### Feature Contribution to Prediction")
    st.dataframe(
        shap_df.sort_values(by="SHAP Impact", key=abs, ascending=False)
    )

    # =========================
    # BAR CHART VIEW (MATPLOTLIB)
    # =========================
    st.write("### Visual Impact View")

    fig, ax = plt.subplots()

    ax.barh(
        shap_df["Feature"],
        shap_df["SHAP Impact"]
    )

    ax.set_xlabel("SHAP Impact")
    ax.set_title("Feature Influence on Prediction")

    st.pyplot(fig)

    # =========================
    # SHAP NATIVE BAR PLOT (CORRECT SHAP VERSION)
    # =========================
    st.write("### SHAP Model View (Advanced)")

    fig2, ax2 = plt.subplots()

    shap.plots.bar(
        shap_vals,
        show=False
    )

    st.pyplot(fig2)

# =========================
# LIME EXPLANATION
# =========================
if st.session_state.prediction_made:

    st.subheader("LIME Explanation")

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
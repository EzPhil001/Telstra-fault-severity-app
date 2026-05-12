# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import uuid
from lime.lime_tabular import LimeTabularExplainer


# =========================
# APP TITLE
# =========================
st.title("Telstra Fault Severity Prediction with XAI")


# =========================
# LOAD MODEL + FEATURES
# =========================
model = joblib.load("xgb_pipeline.pkl")
feature_columns = joblib.load("features.pkl")


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
# INPUT SECTION
# =========================
st.subheader("Enter Fault Features")

log_feature = st.number_input("Log Feature", value=0.0)
event_type = st.number_input("Event Type", value=0.0)
log_volume = st.number_input("Log Volume", value=0.0)
severity_type = st.number_input("Severity Type", value=0.0)
resource_type = st.number_input("Resource Type", value=0.0)


user_input = {
    "log_feature": log_feature,
    "event_type": event_type,
    "log_volume": log_volume,
    "severity_type": severity_type,
    "resource_type": resource_type
}


# enforce correct feature order
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

    # severity rule
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
# SHAP EXPLANATION (FINAL FIXED VERSION)
# =========================
if st.session_state.prediction_made:

    st.subheader("SHAP Explanation")

    # =========================
    # IMPORTANT FIX: USE FULL PIPELINE
    # =========================
    explainer = shap.Explainer(model, input_data)
    shap_values = explainer(input_data)

    # extract SHAP values for first row
    shap_vals = shap_values.values[0]
    shap_vals = np.array(shap_vals).flatten()

    # =========================
    # SAFETY CHECK
    # =========================
    if len(shap_vals) != len(input_data.columns):
        st.error("SHAP mismatch detected (feature alignment issue).")
        st.write("SHAP shape:", shap_vals.shape)
        st.write("Feature count:", len(input_data.columns))
        st.stop()

    # =========================
    # BUILD DATAFRAME
    # =========================
    shap_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Feature Value": input_data.iloc[0].values,
        "SHAP Impact": shap_vals
    })

    st.write("### Feature Contribution Table")

    shap_df = shap_df.reindex(
        shap_df["SHAP Impact"].abs().sort_values(ascending=False).index
    )

    st.dataframe(shap_df)


    # =========================
    # SHAP VISUALIZATION
    # =========================
    st.write("### SHAP Visual Impact")

    fig, ax = plt.subplots()

    ax.barh(
        shap_df["Feature"],
        shap_df["SHAP Impact"]
    )

    ax.set_xlabel("SHAP Impact")
    ax.set_title("Feature Influence on Prediction")

    st.pyplot(fig)


# =========================
# LIME EXPLANATION
# =========================
if st.session_state.prediction_made:

    st.subheader("LIME Explanation")

    # load real training data if available
    try:
        X_train = joblib.load("X_train.pkl")

    except:
        st.warning("X_train.pkl not found. Using fallback approximation.")
        X_train = np.tile(input_data.values, (100, 1))


    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_columns,
        class_names=["0", "1", "2"],
        mode="classification"
    )


    exp = lime_explainer.explain_instance(
        input_data.iloc[0].values,
        model.predict_proba
    )


    lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"])
    st.dataframe(lime_df)


# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Telstra Fault Severity Prediction App | SHAP + LIME Explainability")
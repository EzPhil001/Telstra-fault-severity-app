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
# SHAP EXPLANATION (FINAL FIX)
# =========================
if st.session_state.prediction_made:

    st.subheader("SHAP Explanation")

    # STEP 1: get transformed input from pipeline
    transformed_input = model[:-1].transform(input_data)

    # STEP 2: extract XGBoost model
    xgb_model = model.named_steps["model"]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(transformed_input)

    pred_class = int(st.session_state.prediction)

    # STEP 3: handle multiclass output safely
    if isinstance(shap_values, list):
        shap_vals = shap_values[pred_class]
    else:
        shap_vals = shap_values

    shap_vals = np.array(shap_vals)

    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    shap_vals = shap_vals.flatten()

    # STEP 4: get correct feature names AFTER preprocessing
    feature_names = model[:-1].get_feature_names_out()

    # safety check
    if len(shap_vals) != len(feature_names):
        st.error("SHAP feature mismatch after transformation.")
        st.write("SHAP shape:", shap_vals.shape)
        st.write("Feature count:", len(feature_names))
        st.stop()

    # STEP 5: SHAP dataframe
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Impact": shap_vals
    })

    st.write("### Feature Contribution Table")

    shap_df = shap_df.reindex(
        shap_df["SHAP Impact"].abs().sort_values(ascending=False).index
    )

    st.dataframe(shap_df)

    # STEP 6: visualization
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
# LIME EXPLANATION (FIXED)
# =========================
if st.session_state.prediction_made:

    st.subheader("LIME Explanation")

    # load real training data
    try:
        X_train = joblib.load("X_train.pkl")
    except:
        st.warning("X_train.pkl not found. Using fallback approximation.")
        X_train = np.repeat(input_data.values, 100, axis=0)

    lime_explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_columns,
        class_names=["0", "1", "2"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        input_data.iloc[0].values,
        model.predict_proba,
        num_features=5
    )

    st.write("### LIME Feature Contributions")

    lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Impact"])
    st.dataframe(lime_df)


# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Telstra Fault Severity Prediction App | SHAP + LIME Explainability")
import streamlit as st
import sys
sys.path.append(".")
from core.data import get_patient_raw, get_patient_features
from core.model import predict_risk,load_model
import shap
from explain import generate_reasoning_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Patient Detail", layout="wide")

if "selected_patient" not in st.session_state:
    st.warning("No patient selected. Please go to Patient List first.")
    if st.button("← Go to Patient List"):
        st.switch_page("pages/1_Patient_List.py")
    st.stop()

patient_id = st.session_state["selected_patient"]
st.title(f"🩺 Patient Report — {patient_id}")


raw_df = get_patient_raw(patient_id)
feat_df = get_patient_features(patient_id)
score, label = predict_risk(feat_df)

# Risk header
st.metric(label="Risk Score", value=f"{score}%", delta=label)
st.divider()

# Biomarker trend charts
st.subheader("📈 Biomarker Trends Over Time")

biomarkers = ["ca_125", "glucose", "hemoglobin", "wbc", "bmi"]

cols = st.columns(2)

for i, marker in enumerate(biomarkers):
    with cols[i % 2]:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(raw_df["year"], raw_df[marker], marker="o", linewidth=2)
        ax.set_title(marker.upper().replace("_", " "))
        ax.set_xlabel("Year")
        ax.set_ylabel(marker)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        

st.divider()
st.subheader("🔍 SHAP Feature Explanation")

with st.expander("ℹ️ How to read this chart"):
    st.markdown("""
    **X-axis — SHAP value**
    - Positive (+) → pushes risk **up**
    - Negative (–) → pushes risk **down**

    **Y-axis — Features**
    - Sorted by importance for this specific patient

    **Bar length — Strength of impact**
    - Longer bar = bigger effect on the prediction

    **Color meaning**
    - Pink bars → increase risk (more likely cancer)
    - Blue bars → decrease risk (more likely healthy)
    """)


model = load_model()
explainer = shap.TreeExplainer(model)

FEATURE_COLS = [
    'glucose_first','glucose_mean','glucose_std','glucose_slope',
    'hemoglobin_first','hemoglobin_mean','hemoglobin_std','hemoglobin_slope',
    'wbc_first','wbc_mean','wbc_std','wbc_slope',
    'bmi_first','bmi_mean','bmi_std','bmi_slope',
    'ca_125_first','ca_125_mean','ca_125_std','ca_125_slope'
]

X = feat_df[FEATURE_COLS]
shap_values = explainer.shap_values(X)

fig, ax = plt.subplots(figsize=(8, 4))
shap.bar_plot(shap_values[0], feature_names=FEATURE_COLS, show=False)
st.pyplot(fig)
plt.close(fig)

st.divider()
st.subheader("📋 Reasoning Report")

report = generate_reasoning_report(patient_id)
st.text(report)
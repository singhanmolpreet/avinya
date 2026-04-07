import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import sys
sys.path.append(".")
from core.model import predict_risk
from explain import generate_reasoning_report

st.set_page_config(page_title="Add Patient", layout="wide")
st.title("➕ Add New Patient")
st.markdown("Enter biomarker readings for each visit year.")

patient_id = st.text_input("Patient ID", placeholder="e.g. P201")
n_visits = st.slider(
    "Number of visits",
    min_value=2,
    max_value=10,
    value=5,
    help="Minimum 2 visits needed to compute a trend slope"
)

visit_labels = [f"Visit {i+1}" for i in range(n_visits)]


biomarkers = ["glucose", "hemoglobin", "wbc", "bmi", "ca_125"]

# Build input grid — one row per year, one column per biomarker
st.subheader("📋 Biomarker Readings by Visit Year")

input_data = {}
for marker in biomarkers:
    input_data[marker] = []

cols = st.columns(len(biomarkers) + 1)
cols[0].markdown("**Year**")
for i, marker in enumerate(biomarkers):
    cols[i+1].markdown(f"**{marker.upper()}**")

readings = {marker: [] for marker in biomarkers}

for v_idx, v_label in enumerate(visit_labels):
    cols = st.columns(len(biomarkers) + 1)
    cols[0].markdown(f"**{v_label}**")
    for i, marker in enumerate(biomarkers):
        val = cols[i+1].number_input(
            label=f"{marker}_{v_label}",
            label_visibility="collapsed",
            value=0.0,
            key=f"{marker}_{v_label}"
        )
        readings[marker].append(val)
        
if st.button("🔍 Predict Risk"):
    if not patient_id:
        st.error("Please enter a Patient ID.")
        st.stop()

    # Recreate the same feature engineering as vardaan_features.csv
    features = {"patient_id": patient_id}

    for marker in biomarkers:
        vals = np.array(readings[marker], dtype=float)
        visit_indices = list(range(n_visits))
        slope = linregress(visit_indices, vals).slope()
        features[f"{marker}_first"] = vals[0]
        features[f"{marker}_mean"]  = np.mean(vals)
        features[f"{marker}_std"]   = np.std(vals)
        features[f"{marker}_slope"] = slope

    feat_df = pd.DataFrame([features])

    score, label = predict_risk(feat_df)

    st.divider()
    st.metric("Risk Score", f"{score}%", delta=label)
    
# Temporarily inject into explain.py's dataframe
    import explain as ex
    
    new_row = feat_df.copy()
    new_row["label"] = 0  # placeholder
    
    original_df = ex.df.copy()
    original_X  = ex.X.copy()
    original_sv = ex.shap_values.copy()

    ex.df = pd.concat([ex.df, new_row], ignore_index=True)
    ex.X  = ex.df.drop(columns=["patient_id", "label"])
    ex.shap_values = ex.explainer.shap_values(ex.X)

    st.subheader("📋 Reasoning Report")
    report = generate_reasoning_report(patient_id)
    st.text(report)

    # Restore original state
    ex.df         = original_df
    ex.X          = original_X
    ex.shap_values = original_sv
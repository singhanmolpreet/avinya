import streamlit as st
import sys
sys.path.append(".")
from core.data import load_features
from core.model import predict_all

st.set_page_config(page_title="Patient List", layout="wide")
st.title("📋 Patient List")
df = load_features()
df = predict_all(df)

st.subheader("Search & Filter")

st.subheader("Open Patient Detail")

patient_ids = df["patient_id"].sort_values().tolist()
selected = st.selectbox("Select a Patient ID", patient_ids)

if st.button("View Patient Detail →"):
    st.session_state["selected_patient"] = selected
    st.switch_page("pages/2_Patient_Detail.py")
    

col1, col2 = st.columns([2, 1])

with col1:
    search = st.text_input("Search by Patient ID", placeholder="e.g. P001")

with col2:
    tier_filter = st.selectbox("Filter by Risk Tier",
                           ["HIGH RISK 🔴", "LOW RISK 🟢", "All"])

# Apply filters
if search:
    df = df[df["patient_id"].str.contains(search.upper())]

if tier_filter != "All":
    df = df[df["risk_tier"] == tier_filter]
    

st.subheader(f"Showing {len(df)} patients")

display_cols = ["patient_id", "risk_score", "risk_tier",
                "ca_125_slope", "ca_125_mean", "glucose_slope", "glucose_mean", "bmi_slope", "bmi_mean", "wbc_slope", "wbc_mean"]

st.dataframe(
    df[display_cols].sort_values("risk_score", ascending=False),
    use_container_width=True,
    hide_index=True
)
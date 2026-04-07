import streamlit as st

st.set_page_config(
    page_title="Vardaan AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Vardaan AI")
st.subheader("Early Cancer Risk Prediction System")

st.markdown("""
Welcome to Vardaan AI. Use the sidebar to navigate:
- **Patient List** — view all patients and risk tiers
- **Patient Detail** — deep dive into one patient's history
- **Add Patient** — score a new patient instantly
""")

if st.button("Go to Patient List →"):
        st.switch_page("pages/1_Patient_List.py")
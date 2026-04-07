import pickle
import pandas as pd 
import streamlit as st
import numpy as np

FEATURE_COLS = [
    'glucose_first','glucose_mean','glucose_std','glucose_slope',
    'haemoglobin_first','haemoglobin_mean','haemoglobin_std','haemoglobin_slope',
    'wbc_first','wbc_mean','wbc_std','wbc_slope',
    'bmi_first','bmi_mean','bmi_std','bmi_slope',
    'ca_125_first','ca_125_mean','ca_125_std','ca_125_slope'
]

@st.cache_resource
def load_model():
    with open("vardaan_model.pkl", "rb") as f:
        return pickle.load(f)

def predict_all(features_df):
    model = load_model()
    X = features_df[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    features_df = features_df.copy()
    features_df["risk_score"] = (probs * 100).round(1)
    features_df["risk_tier"] = np.where(probs >= 0.3, "HIGH RISK 🔴", "LOW RISK 🟢")
    return features_df

def predict_risk(patient_features_df):
    model = load_model()
    X = patient_features_df[FEATURE_COLS]
    prob = model.predict_proba(X)[0][1]
    label = "HIGH RISK 🔴" if prob >= 0.3 else "LOW RISK 🟢"
    return round(prob * 100, 1), label
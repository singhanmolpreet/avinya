import pandas as pd
import streamlit as st

@st.cache_data
def load_features():
    df = pd.read_csv("vardaan_features.csv")
    return df

@st.cache_data
def load_raw():
    df = pd.read_csv("vardaan_raw.csv")
    return df

def get_patient_raw(patient_id):
    df = load_raw()
    return df[df["patient_id"] == patient_id].sort_values("year")

def get_patient_features(patient_id):
    df = load_features()
    return df[df["patient_id"] == patient_id]
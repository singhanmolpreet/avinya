import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Vardaan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Metric cards ── */
.metric-card {
    border-radius: 16px;
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.metric-card.total  { background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%); }
.metric-card.high   { background: linear-gradient(135deg, #fff0f0 0%, #ffe0e0 100%); }
.metric-card.low    { background: linear-gradient(135deg, #f0fff4 0%, #d8f5e3 100%); }
.metric-card.rate   { background: linear-gradient(135deg, #fffbf0 0%, #fef3cd 100%); }

.metric-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.55;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    line-height: 1;
    margin: 4px 0 2px;
}
.metric-card.total .metric-value { color: #3b5bdb; }
.metric-card.high  .metric-value { color: #e03131; }
.metric-card.low   .metric-value { color: #2f9e44; }
.metric-card.rate  .metric-value { color: #e67700; }

.metric-sub {
    font-size: 12px;
    opacity: 0.5;
    font-weight: 400;
}

/* ── Hero header ── */
.hero-header {
    padding: 36px 0 8px;
    margin-bottom: 4px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: inherit;
    margin: 0;
}
.hero-sub {
    font-size: 15px;
    opacity: 0.55;
    margin-top: 4px;
    font-weight: 400;
}

/* ── Section title ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    margin: 32px 0 16px;
    opacity: 0.85;
}

/* ── Patient table rows ── */
.risk-high { color: #e03131; font-weight: 600; }
.risk-low  { color: #2f9e44; font-weight: 600; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid rgba(0,0,0,0.07); margin: 24px 0; }

/* ── CTA buttons ── */
.stButton > button {
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 22px;
    border: none;
    transition: all 0.2s ease;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(0,0,0,0.07);
}
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv("vardaan_raw.csv")


def compute_stats(df: pd.DataFrame):
    """Return per-patient summary using the latest year's label."""
    latest = df.sort_values("year").groupby("patient_id").last().reset_index()
    total = len(latest)
    high  = int((latest["label"] == 1).sum())
    low   = total - high
    rate  = round(high / total * 100, 1) if total else 0
    return total, high, low, rate, latest


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    raw = load_raw()
    total, high_risk, low_risk, risk_rate, latest = compute_stats(raw)
    data_ok = True
except FileNotFoundError:
    data_ok  = False
    total = high_risk = low_risk = 0
    risk_rate = 0.0
    latest = pd.DataFrame()


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <p class="hero-title">🩺 Vardaan AI</p>
  <p class="hero-sub">Early Cancer Risk Prediction System &nbsp;·&nbsp; Patient Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card total">
      <div class="metric-label">Total Patients</div>
      <div class="metric-value">{total}</div>
      <div class="metric-sub">Across all years</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card high">
      <div class="metric-label">High Risk</div>
      <div class="metric-value">{high_risk}</div>
      <div class="metric-sub">Label = 1 (cancer risk)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card low">
      <div class="metric-label">Low Risk</div>
      <div class="metric-value">{low_risk}</div>
      <div class="metric-sub">Label = 0 (normal)</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card rate">
      <div class="metric-label">Risk Rate</div>
      <div class="metric-value">{risk_rate}%</div>
      <div class="metric-sub">High-risk proportion</div>
    </div>""", unsafe_allow_html=True)


# ── Quick navigation ──────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Quick Navigation</div>", unsafe_allow_html=True)

n1, n2, n3, _ = st.columns([1, 1, 1, 3])
with n1:
    if st.button("👥  Patient List", use_container_width=True):
        st.switch_page("pages/1_Patient_List.py")
with n2:
    if st.button("🔍  Patient Detail", use_container_width=True):
        st.switch_page("pages/2_Patient_Detail.py")
with n3:
    if st.button("➕  Add Patient", use_container_width=True):
        st.switch_page("pages/3_Add_Patient.py")


# ── Risk breakdown bar ─────────────────────────────────────────────────────────
if data_ok and total > 0:
    st.markdown("<div class='section-title'>Risk Distribution</div>", unsafe_allow_html=True)

    low_pct  = round(low_risk  / total * 100, 1)
    high_pct = round(high_risk / total * 100, 1)

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
      <span style="font-size:13px; font-weight:600; color:#2f9e44; min-width:90px;">Low Risk {low_pct}%</span>
      <div style="flex:1; background:#f0f0f0; border-radius:999px; height:14px; overflow:hidden;">
        <div style="display:flex; height:100%;">
          <div style="width:{low_pct}%; background:linear-gradient(90deg,#40c057,#2f9e44); border-radius:999px 0 0 999px;"></div>
          <div style="width:{high_pct}%; background:linear-gradient(90deg,#fa5252,#e03131); border-radius:0 999px 999px 0;"></div>
        </div>
      </div>
      <span style="font-size:13px; font-weight:600; color:#e03131; min-width:90px; text-align:right;">High Risk {high_pct}%</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Recent high-risk patients ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>⚠️ High-Risk Patients (latest reading)</div>", unsafe_allow_html=True)

    hr_patients = latest[latest["label"] == 1][
        ["patient_id", "year", "glucose", "bmi", "haemoglobin", "wbc", "ca_125"]
    ].sort_values("ca_125", ascending=False).head(10).reset_index(drop=True)

    hr_patients.columns = ["Patient ID", "Year", "Glucose", "BMI", "Haemoglobin", "WBC", "CA-125"]

    st.dataframe(
        hr_patients.style
            .format({"Glucose": "{:.1f}", "BMI": "{:.1f}", "Haemoglobin": "{:.1f}",
                     "WBC": "{:,.0f}", "CA-125": "{:.1f}"})
            .background_gradient(subset=["CA-125"], cmap="Reds")
            .set_properties(**{"font-size": "13px"}),
        use_container_width=True,
        hide_index=True
    )

    # ── Biomarker averages ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Average Biomarkers by Risk Group</div>", unsafe_allow_html=True)

    bio_cols = ["glucose", "bmi", "haemoglobin", "wbc", "ca_125"]
    avg = latest.groupby("label")[bio_cols].mean().round(2)
    avg.index = ["Low Risk (0)", "High Risk (1)"]
    avg.columns = ["Glucose", "BMI", "Haemoglobin", "WBC", "CA-125"]

    st.dataframe(
        avg.style
            .background_gradient(cmap="RdYlGn_r", axis=0)
            .format("{:.2f}")
            .set_properties(**{"font-size": "13px"}),
        use_container_width=True
    )

elif not data_ok:
    st.warning("⚠️ Could not load `vardaan_raw.csv`. Make sure the file is in the project root.", icon="⚠️")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; opacity:0.35; font-size:12px;'>"
    "Vardaan AI · Early Cancer Detection · For clinical use only</p>",
    unsafe_allow_html=True
)
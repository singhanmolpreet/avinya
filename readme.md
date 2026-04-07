# 🩺 Vardaan AI
### Early Cancer Risk Prediction from Longitudinal Blood Test Data

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red) ![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)

---

## The Core Insight

Most clinical ML systems look at a **single snapshot** — one blood test, one reading, one moment in time. Vardaan AI looks at **trends**.

A CA-125 level of 45 U/mL means nothing in isolation. But CA-125 rising from 20 → 28 → 35 → 41 → 45 over five visits — that's a signal. That's what Vardaan AI is designed to catch.

> **Longitudinal slope of biomarkers predicts cancer risk better than any single reading.**

---

## What It Does

Vardaan AI is a B2B clinical decision support tool that:

- Ingests a patient's **visit history** (multiple blood test readings over time)
- Computes **biomarker trends** (slope, mean, std per biomarker)
- Predicts **cancer risk** using a trained XGBoost classifier
- Explains **why** the risk is high or low using SHAP — in plain language a doctor can act on
- Surfaces **high-risk patients first** — so doctors see who needs attention today, not who was seen last

---

## Research Foundation

The biomarker selection and trend logic is grounded in peer-reviewed literature:

| Biomarker | Relevance | Source |
|-----------|-----------|--------|
| **CA-125** | Primary cancer marker; rising trend is a key early signal | Bast et al., 1983 |
| **Glucose** | Elevated fasting glucose linked to cancer incidence | Stattin et al., 2007 |
| **BMI** | Higher BMI associated with increased postmenopausal cancer risk | Renehan et al., 2008 |
| **WBC** | Elevated white blood cell count observed in cancer patients | Noh et al., 2013 |
| **Hemoglobin** | Anaemia patterns associated with malignancy progression | Knight et al., 2004 |

---

## System Architecture

```
vardaan_ai/
│
├── app.py                    ← Entry point and navigation
│
├── pages/
│   ├── 1_Patient_List.py     ← Triage view: all patients, risk badges, search
│   ├── 2_Patient_Detail.py   ← Per-patient: biomarker trends, SHAP, report
│   └── 3_Add_Patient.py      ← New patient form → instant risk score
│
├── core/
│   ├── model.py              ← Load model, batch predict, cache with st.cache_resource
│   └── data.py               ← Load CSVs, filter by patient, cache with st.cache_data
│
├── explain.py                ← SHAP reasoning report generator
├── vardaan_features.csv      ← Engineered features (200 patients × 22 columns)
├── vardaan_raw.csv           ← Raw longitudinal data (1000 rows, 5 visits/patient)
└── vardaan_model.pkl         ← Trained XGBoost classifier
```

---

## ML Pipeline

### 1. Data — `vardaan_raw.csv`
- 200 synthetic patients, 5 visits each = 1,000 rows
- 30% cancer (label=1), 70% healthy (label=0)
- Cancer patients: rising biomarker trends over visits
- Healthy patients: stable biomarker values over visits

### 2. Feature Engineering — `vardaan_features.csv`
For each of 5 biomarkers, four features are computed per patient:

```python
slope, _, _, _, _ = linregress(visit_indices, values)
features[f"{marker}_first"] = values[0]
features[f"{marker}_mean"]  = np.mean(values)
features[f"{marker}_std"]   = np.std(values)
features[f"{marker}_slope"] = slope          ← key signal
```

**22 total features** per patient (5 biomarkers × 4 + patient_id + label).

The slope is the primary signal — it captures the direction of change over time, which a single reading cannot.

### 3. Model — XGBoost Classifier
- Trained on all 200 patients' engineered features
- AUC ≈ 1.0 on synthetic data (expected — the trends are clearly separable by design)
- Top features by SHAP importance: `ca_125_slope`, `bmi_mean`, `glucose_mean`, `wbc_slope`
- **Threshold: 0.3 (not 0.5)** — recall is prioritised over precision for cancer screening. A missed cancer is more dangerous than a false alarm.

### 4. Explainability — SHAP
Each prediction is explained using SHAP TreeExplainer:
- Per-patient bar chart showing which features pushed risk up or down
- Plain-language reasoning report: *"CA-125 Slope raises risk by 61%"*
- Judges and clinicians can audit every prediction

---

## Dashboard Pages

### Patient List
- Loads all 200 patients with pre-computed risk scores
- Defaults to HIGH RISK patients — mimicking a morning triage workflow
- Searchable by patient ID, filterable by risk tier
- Sorted by risk score descending

### Patient Detail
- 5 biomarker trend charts (line plots over visit history)
- SHAP bar chart with guide: what pink/blue bars mean, what bar length means
- Plain-language reasoning report (top 3 contributing features)
- Navigated to via session state — no page reload required

### Add New Patient
- Dynamic visit input grid (2–10 visits, adjustable by slider)
- Feature engineering runs live on submission — same pipeline as training
- Instant risk score + reasoning report
- Minimum 2 visits enforced (mathematical requirement for slope computation)

---

## Setup

```bash
# Clone and enter project
git clone https://github.com/your-username/vardaan-ai.git
cd vardaan-ai

# Install dependencies
pip install streamlit xgboost shap scikit-learn pandas numpy scipy joblib matplotlib

# Run
streamlit run app.py
```

---

## Known Limitations (Synthetic Data)

This version uses a synthetic dataset. We are transparent about what that means:

| Limitation | Explanation |
|------------|-------------|
| AUC ≈ 1.0 | Synthetic trends are mathematically separable — real data will be noisier |
| Fixed visit count in training | Model trained on 5-visit patients; dynamic visits (2–10) may affect slope reliability |
| No appointment data | Real triage would filter by today's scheduled patients, not just risk score |
| `explain.py` recomputes SHAP on every call | Acceptable for demo; production would cache SHAP values per patient |

These are not bugs. They are documented design constraints — and the architecture is built to replace synthetic data with real HMS data when available.

---

## B2B Product Vision

```
Hospital HMS  →  Vardaan API  →  Risk Intelligence Layer
                                        ↓
                              Anonymised risk scores
                                        ↓
                              Insurance partners
```

- Hospitals integrate via API — patient data never leaves their system
- Vardaan returns risk tiers and trend reports
- Anonymised, aggregated risk intelligence is sold to insurers as a data product
- Compliant with longitudinal data privacy standards by design

---

## Built With

- **XGBoost** — gradient boosted trees for tabular clinical data
- **SHAP** — model explainability, per-patient and global
- **Streamlit** — rapid clinical dashboard, no frontend code required
- **SciPy linregress** — slope computation for biomarker trend features
- **Pandas / NumPy** — feature engineering pipeline

---

## Author

Built as part of an applied AI/ML learning project.  
Mentored pipeline: data → features → model → explainability → dashboard.

*Vardaan (वरदान) — a blessing, a boon. Named for what early detection can be.*
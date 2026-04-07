import pandas as pd
import numpy as np
import shap
import joblib

# Load model and features
model = joblib.load('vardaan_model.pkl')
df = pd.read_csv('vardaan_features.csv')

X = df.drop(columns=['patient_id', 'label'])
y = df['label']

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# print(type(shap_values))
# print(shap_values.shape)

def generate_reasoning_report(patient_id):
    idx = df[df['patient_id'] == patient_id].index[0]
    
    risk_score = model.predict_proba(X)[idx][1]
    
    patient_shap = shap_values[idx]
    
    feature_importance = pd.Series(patient_shap, index=X.columns)
    feature_importance = feature_importance.reindex(
        feature_importance.abs().sort_values(ascending=False).index
    )
    
    top3 = feature_importance.head(3)
    total_shap = top3.abs().sum()
    
    lines = []
    lines.append(f"Patient: {patient_id}")
    lines.append(f"Risk Score: {risk_score * 100:.2f}%")
    lines.append(f"Assessment: {'HIGH RISK' if risk_score > 0.3 else 'LOW RISK'}")
    lines.append(f"\nReasoning Report:")
    
    for feature, shap_val in top3.items():
        direction = "raises" if shap_val > 0 else "lowers"
        contribution = round(float((abs(shap_val) / total_shap) * 100), 2)
        readable = feature.replace('_', ' ').replace('ca 125', 'CA-125').title()
        lines.append(f"  - {readable}: {direction} risk by {contribution}%")
    
    return "\n".join(lines)

if __name__ == "__main__":
    print(generate_reasoning_report('P001'))
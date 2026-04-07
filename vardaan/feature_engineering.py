import pandas as pd
import numpy as np
from scipy.stats import linregress

df=pd.read_csv("vardaan_raw.csv")

# print(df.head(3))
biomarker=["glucose","bmi","haemoglobin", "wbc","ca_125"]

rows=[]

for pid, group in df.groupby("patient_id"):
    row={}
    row["patient_id"]=pid
    row["label"]=group["label"].values[0]
    
    visits=list(range(len(group)))
    for bio in biomarker:
        row[f"{bio}_first"]=round(group[bio].values[0],4)
        row[f"{bio}_mean"]=round(np.mean(group[bio].values),4)
        row[f"{bio}_std"]=round(np.std(group[bio].values),4)
        row[f"{bio}_slope"]=round(linregress(visits,group[bio]).slope, 4)
    
    rows.append(row)
    
features_df=pd.DataFrame(rows)

df.to_csv("vardaan_features.csv")

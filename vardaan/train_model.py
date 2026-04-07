from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("vardaan_features.csv")

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

X=df.drop(columns=["patient_id","label"])
Y=df["label"]

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(X_test)
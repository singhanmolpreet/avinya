from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

df=pd.read_csv("vardaan_features.csv")


X=df.drop(columns=["patient_id","label"])
Y=df["label"]


X_train, X_test, Y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# print(X_train.shape)
# print(X_test.shape)
# print(X_test)

model = XGBClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=2,
    gamma=1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC Score: {round(roc_auc_score(y_test, y_prob), 3)}")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")

joblib.dump(model, 'vardaan_model.pkl')
# print("Model saved.")
    
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().tail(10).plot(kind='barh', figsize=(8,5))
plt.title('Top 10 most important features')
plt.tight_layout()
plt.show()

# print(df.groupby('label')[['wbc_first', 'ca_125_slope', 'bmi_slope']].mean())

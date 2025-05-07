# Dataiku-Style Insurance Fraud Detection Pipeline
# Author: Roman Dobczansky

# === Setup ===
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === Step 1: Upload Data ===
from google.colab import files
uploaded = files.upload()

# === Step 2: Load and Clean Data ===
df = pd.read_csv("insurance_claims.csv")
df.dropna(inplace=True)
df = df[df['claim_amount'] > 0]

# === Step 3: Feature Engineering ===
df['claim_ratio'] = df['claim_amount'] / df['annual_income']
df['is_high_claim'] = df['claim_ratio'] > 0.5
df['label'] = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)

# === Step 4: Model Training ===
X = df[['claim_ratio']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Step 5: Evaluation ===
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()

# === Step 6: Save Model ===
joblib.dump(clf, "fraud_model.pkl")

# === Step 7: Predict New Claim ===
def predict_new_claim(claim_ratio):
    model = joblib.load("fraud_model.pkl")
    return model.predict([[claim_ratio]])[0]

# Example
example_ratio = 0.6
print(f"Predicted Fraud (1=Yes, 0=No) for claim ratio {example_ratio}: {predict_new_claim(example_ratio)}")

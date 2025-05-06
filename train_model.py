# scripts/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from scripts.preprocessing import load_and_clean_data
from scripts.features import add_features

# Load and preprocess data
df = load_and_clean_data('data/insurance_claims.csv')
df = add_features(df)

# Define target and features
X = df[['claim_ratio']]
y = df['fraud_reported'].apply(lambda x: 1 if x == 'Y' else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'models/fraud_model.pkl')

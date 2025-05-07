# === Setup ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# === Step 1: Upload Data ===
from google.colab import files
uploaded = files.upload()

# === Step 2: Load and Preprocess ===
df = pd.read_csv("insurance_claims.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Columns:", df.columns.tolist())

# === Step 3: Feature Engineering ===
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
X = df.drop(columns=['charges'])
y = df['charges']

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: Model Training ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 6: Evaluation ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# === Step 7: Save Model ===
joblib.dump(model, "insurance_cost_predictor.pkl")

# === Step 8: Predict New Data ===
def predict_insurance_cost(age, bmi, children, sex, smoker, region):
    input_data = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
    }])
    model = joblib.load("insurance_cost_predictor.pkl")
    return model.predict(input_data)[0]

# Example Prediction
cost = predict_insurance_cost(35, 27.5, 2, 'male', 'no', 'northwest')
print(f"Predicted Insurance Cost: ${cost:.2f}")

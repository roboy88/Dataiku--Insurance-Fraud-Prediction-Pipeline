# scripts/predict.py
import pandas as pd
import joblib

def predict_new_claim(claim_ratio):
    model = joblib.load('models/fraud_model.pkl')
    return model.predict([[claim_ratio]])[0]

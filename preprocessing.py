# scripts/preprocessing.py
import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df = df[df['claim_amount'] > 0]
    return df

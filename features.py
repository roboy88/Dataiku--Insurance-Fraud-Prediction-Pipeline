# scripts/features.py
def add_features(df):
    df['claim_ratio'] = df['claim_amount'] / df['annual_income']
    df['is_high_claim'] = df['claim_ratio'] > 0.5
    return df

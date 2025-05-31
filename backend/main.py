from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
app = FastAPI()

def preprocess_df(df):
    """
    Drop unused columns and add ratio/log features.
    """
    eps = 1e-6
    df = df.copy()
    # Drop
    df.drop(columns=['zip_count_4w', 'bank_branch_count_8w'], inplace=True, errors='ignore')
    # Ratios
    df['age_month_ratio'] = np.where(df['prev_address_months_count'] > 0,
                                     df['customer_age'] / (df['prev_address_months_count'] + eps), 0.0)
    df['score_age_ratio'] = np.where(df['customer_age'] > 0,
                                     df['credit_risk_score'] / (df['customer_age'] + eps), 0.0)
    if 'device_distinct_emails_8w' in df and 'date_of_birth_distinct_emails_4w' in df:
        df['email_growth'] = np.where(df['date_of_birth_distinct_emails_4w'] > 0,
                                      df['device_distinct_emails_8w'] / (df['date_of_birth_distinct_emails_4w'] + eps), 0.0)
    # Log transforms
    for col in ['velocity_6h', 'velocity_24h', 'velocity_4w']:
        if col in df:
            df[f'log_{col}'] = np.log1p(df[col])
    # Session bins
    df['sess_bin_short'] = (df['session_length_in_minutes'] < 5).astype(int)
    df['sess_bin_long'] = (df['session_length_in_minutes'] > 30).astype(int)
    return df

#-----
import sys, types
fake = types.ModuleType("__main__")
fake.preprocess_df = preprocess_df
sys.modules["__main__"] = fake
#-----
import joblib
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = '../model/models/fraud_detector.pkl'
try:
    loaded = joblib.load(model_path)
    model = loaded['model']
    threshold = loaded['threshold']
except Exception as e:
    raise RuntimeError(f"failed to load the model from {model_path}: {e}")


class ModelInput(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    device_distinct_emails_8w: int
    employment_status: str
    credit_risk_score: int
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: int
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int
    payment_type: str
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int
    device_fraud_count: int
    month: int

@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        df = pd.DataFrame([input_data.dict()])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error transforming data: {e}")

    expected_cols = set(model.named_steps['step2'].transformers_[0][2] +
                        model.named_steps['step2'].transformers_[1][2] +
                        ['session_length_in_minutes', 'zip_count_4w', 'bank_branch_count_8w',
                         'prev_address_months_count', 'device_distinct_emails_8w',
                         'credit_risk_score'])
    try:
        probs = model.predict_proba(df)[:, 1]
        pred_label = (probs >= threshold).astype(int)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {
        "prediction": int(pred_label),
        "probability": float(probs[0]),
        "threshold_used": threshold
    }

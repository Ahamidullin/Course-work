import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    RandomizedSearchCV,
    cross_val_predict
)
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
import joblib
import matplotlib.pyplot as plt

# Constants
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.25
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load raw data
df = pd.read_csv("Data/Base.csv")

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

# Preprocessing transformers
step1 = FunctionTransformer(preprocess_df)

# Define numeric and categorical columns for ColumnTransformer
# We apply step1 to df first to get accurate columns
df_temp = preprocess_df(df)
num_cols = df_temp.select_dtypes(include=[np.number]).columns.drop('fraud_bool').tolist()
cat_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']

step2 = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

# Define stacking model (must be before pipeline)
estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE))
]
final_est = LGBMClassifier(random_state=RANDOM_STATE)
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=final_est,
    passthrough=True,
    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
)

# Build full pipeline: preprocess -> balance -> model
pipeline = Pipeline([
    ('step1', step1),
    ('step2', step2),
    ('smote', SMOTEENN(random_state=RANDOM_STATE)),
    ('stack', stack)
])

# Split data
y = df['fraud_bool']
X = df.drop(columns=['fraud_bool'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# Compute and set scale_pos_weight for final LGBM estimator
pos_w = (y_train == 0).sum() / (y_train == 1).sum()
pipeline.named_steps['stack'].final_estimator.set_params(scale_pos_weight=pos_w)

# Hyperparameter search for final estimator inside stacking
param_dist = {
    'stack__final_estimator__num_leaves': [31, 50, 70],
    'stack__final_estimator__learning_rate': [0.01, 0.05, 0.1],
    'stack__final_estimator__min_data_in_leaf': [20, 50, 100],
    'stack__final_estimator__feature_fraction': [0.6, 0.8, 1.0]
}
search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=20,
    scoring='average_precision',
    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
    n_jobs=-1,
    random_state=RANDOM_STATE
)
search.fit(X_train, y_train)
best_pipe = search.best_estimator_

# OOF predictions for threshold selection
probs_oof = cross_val_predict(
    best_pipe,
    X_train,
    y_train,
    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
    method='predict_proba',
    n_jobs=-1
)[:, 1]
prec, rec, th = precision_recall_curve(y_train, probs_oof)
best_t = th[np.argmax(2 * prec * rec / (prec + rec + 1e-8))]

# Final fit & save
best_pipe.fit(X_train, y_train)
joblib.dump({'model': best_pipe, 'threshold': best_t}, 'models/fraud_detector.pkl')

# Test evaluation
y_pred = (best_pipe.predict_proba(X_test)[:,1] >= best_t).astype(int)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:,1]))
print("PR AUC:", average_precision_score(y_test, best_pipe.predict_proba(X_test)[:,1]))

# Plot PR curve
prp, prr, _ = precision_recall_curve(y_test, best_pipe.predict_proba(X_test)[:,1])
plt.figure(figsize=(8,6))
plt.plot(prr, prp, label=f"AP={average_precision_score(y_test, best_pipe.predict_proba(X_test)[:,1]):.3f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('plots/pr_curve_test.png')
plt.close()

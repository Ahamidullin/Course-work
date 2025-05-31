import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
import category_encoders as ce
import joblib
import matplotlib.pyplot as plt

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.25
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("Data/Base.csv")

def safe_ratio(numerator, denominator, eps=1e-6):
    return np.where(denominator > 0, numerator / (denominator + eps), 0.0)

df['age_month_ratio'] = safe_ratio(df['customer_age'], df['prev_address_months_count'])
df['score_age_ratio'] = safe_ratio(df['credit_risk_score'], df['customer_age'])
if 'device_distinct_emails_8w' in df and 'date_of_birth_distinct_emails_4w' in df:
    df['email_growth'] = safe_ratio(df['device_distinct_emails_8w'], df['date_of_birth_distinct_emails_4w'])
for col in ['velocity_6h', 'velocity_24h', 'velocity_4w']:
    if col in df:
        df[f'log_{col}'] = np.log1p(df[col])
df['sess_bin_short'] = (df['session_length_in_minutes'] < 5).astype(int)
df['sess_bin_long'] = (df['session_length_in_minutes'] > 30).astype(int)

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

cat_cols = ['source', 'payment_type', 'device_os']
for col in cat_cols:
    if col in df:
        encoder = ce.TargetEncoder(cols=[col])
        df[col] = encoder.fit_transform(df[col], df['fraud_bool'])

y = df['fraud_bool']
X = df.drop(columns=['fraud_bool', 'zip_count_4w', 'bank_branch_count_8w'], errors='ignore')
X = pd.get_dummies(X, drop_first=True)
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if (upper[c] > 0.9).any()]
X.drop(columns=to_drop, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

smote = SMOTEENN(random_state=RANDOM_STATE)
estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE))
]
scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
final = LGBMClassifier(scale_pos_weight=scale_pw, random_state=RANDOM_STATE)
stack = StackingClassifier(estimators=estimators, final_estimator=final, passthrough=True, cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE))
pipeline = Pipeline([('balance', smote), ('model', stack)])

param_dist = {
    'model__final_estimator__num_leaves': [31, 50, 70],
    'model__final_estimator__learning_rate': [0.01, 0.05, 0.1],
    'model__final_estimator__min_data_in_leaf': [20, 50, 100],
    'model__final_estimator__feature_fraction': [0.6, 0.8, 1.0]
}
search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, scoring='average_precision', cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1, random_state=RANDOM_STATE)
search.fit(X_train, y_train)
best_pipe = search.best_estimator_

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
probs_oof = cross_val_predict(best_pipe, X_train, y_train, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
prec, rec, th = precision_recall_curve(y_train, probs_oof)
f1 = 2 * prec * rec / (prec + rec + 1e-8)
best_t = th[np.argmax(f1)]
print(f"Optimal threshold: {best_t:.4f}")
y_pred_train = (probs_oof >= best_t).astype(int)
print(classification_report(y_train, y_pred_train))

best_pipe.fit(X_train, y_train)
joblib.dump({'model': best_pipe, 'threshold': best_t}, 'models/fraud_detector.pkl')

probs_test = best_pipe.predict_proba(X_test)[:, 1]
y_pred = (probs_test >= best_t).astype(int)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(roc_auc_score(y_test, probs_test))
print(average_precision_score(y_test, probs_test))

tp, tr, _ = precision_recall_curve(y_test, probs_test)
plt.figure(figsize=(8, 6))
plt.plot(tr, tp, label=f'AP={average_precision_score(y_test, probs_test):.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('plots/pr_curve_test.png')
plt.close()
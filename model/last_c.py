import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    average_precision_score, precision_recall_curve,
    make_scorer
)

from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Data/Base.csv')
target_col = 'fraud_bool'
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

def remove_outliers_iqr(df, numeric_cols):
    df_clean = df.copy()
    for col in numeric_cols:
        q1, q3 = df_clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

df_clean = remove_outliers_iqr(df, numeric_cols)
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

if categorical_cols:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    train_cat = pd.DataFrame(
        encoder.fit_transform(X_train[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_train.index
    )
    test_cat = pd.DataFrame(
        encoder.transform(X_test[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_test.index
    )
    X_train = pd.concat([X_train.drop(columns=categorical_cols), train_cat], axis=1)
    X_test = pd.concat([X_test.drop(columns=categorical_cols), test_cat], axis=1)

sme = SMOTEENN(random_state=42)
X_train_bal, y_train_bal = sme.fit_resample(X_train, y_train)
print("After SMOTEENN, train class counts:", np.bincount(y_train_bal))

recall_scorer = make_scorer(recall_score)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'BalancedRandomForest': BalancedRandomForestClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

params = {
    'LogisticRegression': {'C': [0.01, 0.1, 1.0]},
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'BalancedRandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'LightGBM': {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01],
        'scale_pos_weight': [ (y_train_bal==0).sum() / (y_train_bal==1).sum() ]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01],
        'scale_pos_weight': [ (y_train_bal==0).sum() / (y_train_bal==1).sum() ]
    },
    'CatBoost': {
        'iterations': [100, 200],
        'learning_rate': [0.1, 0.01],
        'auto_class_weights': ['Balanced']
    }
}

best_models = {}
for name, clf in models.items():
    grid = GridSearchCV(clf, params[name], cv=5, scoring=recall_scorer, n_jobs=-1)
    grid.fit(X_train_bal, y_train_bal)
    best_models[name] = grid.best_estimator_
    print(f"{name} best params: {grid.best_params_}")

best_thresholds = {}
for name, model in best_models.items():
    y_scores = model.predict_proba(X_test)[:, 1]
    _, recall_vals, thresholds = precision_recall_curve(y_test, y_scores)
    best_idx = np.argmax(recall_vals)
    best_thresholds[name] = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"{name} threshold for max recall: {best_thresholds[name]:.3f}")

results = []
for name, model in best_models.items():
    thresh = best_thresholds[name]
    y_pred = (model.predict_proba(X_test)[:, 1] >= thresh).astype(int)
    results.append({
        'Model': name,
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC-PR': average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    })

metrics_df = pd.DataFrame(results).sort_values('Recall', ascending=False)
print(metrics_df)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, metric in zip(axes, ['Recall', 'Precision', 'F1', 'AUC-PR']):
    sns.barplot(x='Model', y=metric, data=metrics_df, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(metric)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()

first_clf = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    scale_pos_weight=(y_train_bal==0).sum()/(y_train_bal==1).sum(),
    random_state=42
)
first_clf.fit(X_train_bal, y_train_bal)

first_thresh = 0.3
first_scores = first_clf.predict_proba(X_test)[:, 1]
mask_suspicious = first_scores >= first_thresh

X2 = X_test[mask_suspicious]
y2 = y_test[mask_suspicious]

second_clf = CatBoostClassifier(
    iterations=200,
    learning_rate=0.01,
    auto_class_weights='Balanced',
    random_state=42,
    verbose=0
)
second_clf.fit(X_train_bal, y_train_bal)

scores2 = second_clf.predict_proba(X2)[:, 1]
prec2, rec2, thresholds2 = precision_recall_curve(y2, scores2)

valid_mask = rec2 >= 0.95

if valid_mask.any():
    valid_prec = prec2[valid_mask]
    valid_thresholds = thresholds2[valid_mask[:len(thresholds2)]]
    best_idx2 = np.argmax(valid_prec)
    best_thresh2 = valid_thresholds[best_idx2]
    print(f"Found threshold with recall ≥ 0.95: {best_thresh2:.4f}, precision = {valid_prec[best_idx2]:.4f}")
else:
    f1s2 = 2 * (prec2 * rec2) / (prec2 + rec2 + 1e-6)
    best_idx2 = np.argmax(f1s2)
    best_thresh2 = thresholds2[best_idx2]
    print(f"!!! No threshold with recall ≥ 0.95 — fallback to best F1: {best_thresh2:.4f}")

final_pred = np.zeros_like(y_test)
final_pred[mask_suspicious] = (scores2 >= best_thresh2).astype(int)

print("\nTwo-stage pipeline metrics:")
print(f"Threshold used (2nd stage): {best_thresh2:.4f}")
print("Recall:", recall_score(y_test, final_pred))
print("Precision:", precision_score(y_test, final_pred))
print("F1:", f1_score(y_test, final_pred))

if valid_mask.any():
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds2[valid_mask[:len(thresholds2)]], prec2[valid_mask], label='Precision (recall ≥ 0.95)')
    plt.axvline(best_thresh2, color='red', linestyle='--', label=f'Selected Threshold = {best_thresh2:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Second Stage: Precision vs Threshold (recall ≥ 0.95)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

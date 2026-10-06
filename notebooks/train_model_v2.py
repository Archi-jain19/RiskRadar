"""
DropGuard AI — Optimized Model Training v2
===========================================
Improvements over v1:
1. More aggressive feature engineering (interaction features, polynomial)
2. Hyperparameter tuning with RandomizedSearchCV
3. Optimized classification threshold
4. Feature selection via importance filtering
5. Better SMOTE strategy (SMOTEENN — combines oversampling + cleaning)
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print("  DropGuard AI — Optimized Model Training v2")
print("=" * 65)

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA & ADD INTERACTION FEATURES
# ─────────────────────────────────────────────────────────

print("\n[1/7] Loading data & engineering interaction features...")

enriched = pd.read_csv(os.path.join(DATA_DIR, 'enriched_dataset.csv'))

# --- NEW interaction features ---
# GPA x Attendance interaction (students with both low GPA AND low attendance are high risk)
enriched['gpa_x_attendance'] = enriched['GPA'] * enriched['Attendance_Rate'] / 100

# Stress x Study inverse (high stress + low study = danger)
enriched['stress_study_ratio'] = enriched['Stress_Index'] / (enriched['Study_Hours_per_Day'] + 0.1)

# Academic momentum (are they getting better or worse?)
enriched['academic_momentum'] = enriched['Semester_GPA'] - enriched['GPA']

# Travel burden impact on performance
enriched['travel_gpa_impact'] = enriched['Travel_Time_Minutes'] / (enriched['GPA'] + 0.1)

# Financial + Academic risk compound
enriched['financial_academic_risk'] = enriched['financial_stress'] * enriched['academic_risk_score']

# Delay severity (delay relative to attendance — chronic avoider?)
enriched['delay_severity'] = enriched['Assignment_Delay_Days'] * (1 - enriched['Attendance_Rate'] / 100)

# Scholarship benefit (does scholarship + study hours lead to better GPA?)
enriched['scholarship_efficiency'] = enriched['Scholarship'] * enriched['GPA']

# Age risk factor (older students may have different dropout patterns)
enriched['age_normalized'] = (enriched['Age'] - enriched['Age'].mean()) / enriched['Age'].std()

new_feats = ['gpa_x_attendance', 'stress_study_ratio', 'academic_momentum',
             'travel_gpa_impact', 'financial_academic_risk', 'delay_severity',
             'scholarship_efficiency', 'age_normalized']
print(f"  ✓ Added {len(new_feats)} interaction features")
print(f"  ✓ Total features: {len(enriched.columns) - 1}")

# ─────────────────────────────────────────────────────────
# 2. SPLIT & BALANCE
# ─────────────────────────────────────────────────────────

print("\n[2/7] Splitting & balancing data...")

from sklearn.model_selection import train_test_split

X = enriched.drop(columns=['Dropout'])
y = enriched['Dropout']
feature_names = list(X.columns)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Try SMOTEENN (SMOTE + Edited Nearest Neighbors) for cleaner boundaries
try:
    smoteenn = SMOTEENN(random_state=42)
    X_train_bal, y_train_bal = smoteenn.fit_resample(X_train, y_train)
    print(f"  ✓ SMOTEENN applied: {len(X_train)} → {len(X_train_bal)} samples")
except:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"  ✓ SMOTE fallback: {len(X_train)} → {len(X_train_bal)} samples")

print(f"    Not Dropout: {(y_train_bal == 0).sum()} | Dropout: {(y_train_bal == 1).sum()}")

# Scale
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=feature_names)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# ─────────────────────────────────────────────────────────
# 3. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────

print("\n[3/7] Hyperparameter tuning (this may take 2-3 minutes)...")

# XGBoost tuning
xgb_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0],
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    xgb_params, n_iter=40, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X_train_scaled, y_train_bal)
xgb_best = xgb_search.best_estimator_
print(f"  ✓ XGBoost best CV AUC: {xgb_search.best_score_:.4f}")
print(f"    Params: depth={xgb_best.max_depth}, lr={xgb_best.learning_rate}, n={xgb_best.n_estimators}")

# LightGBM tuning
lgbm_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [4, 6, 8, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_samples': [5, 10, 20],
    'num_leaves': [20, 31, 50, 70],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1.0, 2.0],
}

lgbm_search = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    lgbm_params, n_iter=40, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1, verbose=0
)
lgbm_search.fit(X_train_scaled, y_train_bal)
lgbm_best = lgbm_search.best_estimator_
print(f"  ✓ LightGBM best CV AUC: {lgbm_search.best_score_:.4f}")

# Random Forest tuning
rf_params = {
    'n_estimators': [200, 300, 500],
    'max_depth': [8, 12, 16, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    rf_params, n_iter=30, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train_scaled, y_train_bal)
rf_best = rf_search.best_estimator_
print(f"  ✓ Random Forest best CV AUC: {rf_search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────
# 4. BUILD OPTIMIZED ENSEMBLE
# ─────────────────────────────────────────────────────────

print("\n[4/7] Building optimized stacking ensemble...")

ensemble = StackingClassifier(
    estimators=[
        ('xgboost', xgb_best),
        ('random_forest', rf_best),
        ('lightgbm', lgbm_best),
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1,
    passthrough=False
)

ensemble.fit(X_train_scaled, y_train_bal)
print("  ✓ Ensemble trained")

# ─────────────────────────────────────────────────────────
# 5. OPTIMIZE CLASSIFICATION THRESHOLD
# ─────────────────────────────────────────────────────────

print("\n[5/7] Optimizing classification threshold...")

# Use validation set to find optimal threshold
y_val_proba = ensemble.predict_proba(X_val_scaled)[:, 1]

# Find threshold that maximizes F1
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]

print(f"  ✓ Default threshold: 0.50")
print(f"  ✓ Optimal threshold: {optimal_threshold:.4f}")

# Compare default vs optimal on validation
y_val_pred_default = (y_val_proba >= 0.5).astype(int)
y_val_pred_optimal = (y_val_proba >= optimal_threshold).astype(int)

print(f"\n  Validation — Default (0.5):")
print(f"    Accuracy: {accuracy_score(y_val, y_val_pred_default):.4f} | F1: {f1_score(y_val, y_val_pred_default):.4f}")
print(f"  Validation — Optimal ({optimal_threshold:.4f}):")
print(f"    Accuracy: {accuracy_score(y_val, y_val_pred_optimal):.4f} | F1: {f1_score(y_val, y_val_pred_optimal):.4f}")

# ─────────────────────────────────────────────────────────
# 6. EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────

print("\n[6/7] Evaluating on test set...")

y_test_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
y_test_pred_default = ensemble.predict(X_test_scaled)
y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

# Metrics with default threshold
metrics_default = {
    'accuracy': accuracy_score(y_test, y_test_pred_default),
    'precision': precision_score(y_test, y_test_pred_default),
    'recall': recall_score(y_test, y_test_pred_default),
    'f1_score': f1_score(y_test, y_test_pred_default),
    'auc_roc': roc_auc_score(y_test, y_test_proba)
}

# Metrics with optimal threshold
metrics_optimal = {
    'accuracy': accuracy_score(y_test, y_test_pred_optimal),
    'precision': precision_score(y_test, y_test_pred_optimal),
    'recall': recall_score(y_test, y_test_pred_optimal),
    'f1_score': f1_score(y_test, y_test_pred_optimal),
    'auc_roc': roc_auc_score(y_test, y_test_proba)
}

print("\n  ┌──────────────────────────────────────────────────────────┐")
print("  │              TEST SET COMPARISON                        │")
print("  ├─────────────────┬───────────────────┬───────────────────┤")
print("  │ Metric          │ v1 (Previous)     │ v2 (Optimized)    │")
print("  ├─────────────────┼───────────────────┼───────────────────┤")
v1 = {'accuracy': 0.7873, 'precision': 0.5483, 'recall': 0.5467, 'f1_score': 0.5475, 'auc_roc': 0.8045}
for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
    old = v1[metric]
    new = metrics_optimal[metric]
    diff = new - old
    arrow = "📈" if diff > 0 else ("📉" if diff < 0 else "━━")
    print(f"  │ {metric:15s} │ {old:.4f}            │ {new:.4f} ({diff:+.4f}) {arrow}│")
print("  └─────────────────┴───────────────────┴───────────────────┘")

print(f"\n  Classification Report (Optimized Threshold = {optimal_threshold:.4f}):")
print(classification_report(y_test, y_test_pred_optimal, target_names=['Not Dropout', 'Dropout']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_optimal)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Dropout', 'Dropout'],
            yticklabels=['Not Dropout', 'Dropout'])
plt.title(f'Confusion Matrix — Optimized Ensemble (threshold={optimal_threshold:.3f})',
          fontsize=13, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix_v2.png'), dpi=150)
print(f"  ✓ Saved confusion_matrix_v2.png")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_best.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(data=importance.head(25), x='importance', y='feature', palette='viridis')
plt.title('Top 25 Feature Importances (Tuned XGBoost v2)', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'feature_importance_v2.png'), dpi=150)
print(f"  ✓ Saved feature_importance_v2.png")

# ─────────────────────────────────────────────────────────
# 7. SHAP + SAVE
# ─────────────────────────────────────────────────────────

print("\n[7/7] SHAP explanations & saving...")

try:
    import shap
    explainer = shap.TreeExplainer(xgb_best)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names,
                      show=False, max_display=25)
    plt.title('SHAP Feature Impact (v2)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'shap_summary_v2.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved shap_summary_v2.png")
    
    joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer.joblib'))
except Exception as e:
    print(f"  ⚠ SHAP: {e}")

# Save all artifacts
joblib.dump(ensemble, os.path.join(MODELS_DIR, 'ensemble_model.joblib'))
joblib.dump(xgb_best, os.path.join(MODELS_DIR, 'xgboost_model.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
joblib.dump(optimal_threshold, os.path.join(MODELS_DIR, 'optimal_threshold.joblib'))

# Save enriched dataset with new features
enriched.to_csv(os.path.join(DATA_DIR, 'enriched_dataset.csv'), index=False)

# Save updated metrics
with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
    json.dump({
        'version': 'v2_optimized',
        'optimal_threshold': float(optimal_threshold),
        'test_metrics_default_threshold': metrics_default,
        'test_metrics_optimal_threshold': metrics_optimal,
        'improvement_over_v1': {k: float(metrics_optimal[k] - v1[k]) for k in v1},
        'tuning': {
            'xgb_best_cv_auc': float(xgb_search.best_score_),
            'lgbm_best_cv_auc': float(lgbm_search.best_score_),
            'rf_best_cv_auc': float(rf_search.best_score_),
        }
    }, f, indent=2)

print(f"  ✓ All artifacts saved to {MODELS_DIR}")

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  MODEL v2 TRAINING COMPLETE ✓")
print("=" * 65)
best = metrics_optimal
print(f"""
  Optimized Ensemble Results:
  ├── Accuracy:   {best['accuracy']:.4f}  (was 0.7873)
  ├── Precision:  {best['precision']:.4f}  (was 0.5483)
  ├── Recall:     {best['recall']:.4f}  (was 0.5467)
  ├── F1-Score:   {best['f1_score']:.4f}  (was 0.5475)
  ├── AUC-ROC:    {best['auc_roc']:.4f}  (was 0.8045)
  └── Threshold:  {optimal_threshold:.4f}  (was 0.5000)
""")

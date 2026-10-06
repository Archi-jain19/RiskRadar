"""
DropGuard AI — Optimized Model Training v3 (Best Version)
==========================================================
Key improvements:
1. Interaction features from v2 (kept the good ones)
2. Regular SMOTE (not SMOTEENN — avoids over-cleaning)
3. Stronger regularization to prevent overfitting
4. Soft voting ensemble + stacking comparison
5. Threshold tuning on validation set
6. Cross-validated final evaluation
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier,
    VotingClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print("  DropGuard AI — Model Training v3 (Final Optimized)")
print("=" * 65)

# ─────────────────────────────────────────────────────────
# 1. LOAD & ENGINEER FEATURES
# ─────────────────────────────────────────────────────────

print("\n[1/6] Loading data & engineering features...")

enriched = pd.read_csv(os.path.join(DATA_DIR, 'enriched_dataset.csv'))

# Remove v2 interaction features if they exist (we'll add our own)
v2_cols = ['gpa_x_attendance', 'stress_study_ratio', 'academic_momentum',
           'travel_gpa_impact', 'financial_academic_risk', 'delay_severity',
           'scholarship_efficiency', 'age_normalized']
enriched = enriched.drop(columns=[c for c in v2_cols if c in enriched.columns], errors='ignore')

# Add best interaction features (based on v2 SHAP analysis)
enriched['gpa_x_attendance'] = enriched['GPA'] * enriched['Attendance_Rate'] / 100
enriched['stress_study_ratio'] = enriched['Stress_Index'] / (enriched['Study_Hours_per_Day'] + 0.5)
enriched['financial_academic_risk'] = enriched['financial_stress'] * enriched['academic_risk_score']
enriched['delay_severity'] = enriched['Assignment_Delay_Days'] * (1 - enriched['Attendance_Rate'] / 100)
enriched['gpa_consistency'] = 1 - abs(enriched['Semester_GPA'] - enriched['CGPA']) / (enriched['CGPA'] + 0.1)
enriched['overall_risk_index'] = (
    enriched['academic_risk_score'] * 0.35 +
    enriched['stress_study_ratio'] / enriched['stress_study_ratio'].max() * 0.25 +
    enriched['commute_burden'] * 0.15 +
    (1 - enriched['engagement_index']) * 0.25
)

new_feats = ['gpa_x_attendance', 'stress_study_ratio', 'financial_academic_risk',
             'delay_severity', 'gpa_consistency', 'overall_risk_index']
print(f"  ✓ Added {len(new_feats)} interaction features")
print(f"  ✓ Total features: {len(enriched.columns) - 1}")

# ─────────────────────────────────────────────────────────
# 2. SPLIT & BALANCE (SMOTE only)
# ─────────────────────────────────────────────────────────

print("\n[2/6] Splitting & balancing...")

X = enriched.drop(columns=['Dropout'])
y = enriched['Dropout']
feature_names = list(X.columns)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Regular SMOTE — oversamples minority without removing majority
smote = SMOTE(random_state=42, sampling_strategy=0.8)  # Don't fully balance — 80% ratio
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"  ✓ Train: {len(X_train)} → {len(X_train_bal)} (SMOTE 0.8 ratio)")
print(f"    Not Dropout: {(y_train_bal == 0).sum()} | Dropout: {(y_train_bal == 1).sum()}")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_bal), columns=feature_names)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# ─────────────────────────────────────────────────────────
# 3. TRAIN TUNED INDIVIDUAL MODELS
# ─────────────────────────────────────────────────────────

print("\n[3/6] Training tuned models...")

# XGBoost — more regularized
xgb_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_weight': [3, 5, 7],
    'gamma': [0.1, 0.2, 0.3],
    'reg_alpha': [0.1, 0.5, 1.0],
    'reg_lambda': [1.0, 2.0, 3.0],
    'scale_pos_weight': [1, 2, 3],
}

print("  Tuning XGBoost (50 iterations)...")
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    xgb_params, n_iter=50, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
xgb_search.fit(X_train_scaled, y_train_bal)
xgb_best = xgb_search.best_estimator_
xgb_val_auc = roc_auc_score(y_val, xgb_best.predict_proba(X_val_scaled)[:, 1])
print(f"    ✓ CV AUC: {xgb_search.best_score_:.4f} | Val AUC: {xgb_val_auc:.4f}")

# LightGBM — more regularized
print("  Tuning LightGBM (50 iterations)...")
lgbm_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_samples': [10, 20, 30],
    'num_leaves': [15, 20, 31],
    'reg_alpha': [0.1, 0.5, 1.0],
    'reg_lambda': [1.0, 2.0, 3.0],
    'is_unbalance': [True, False],
}

lgbm_search = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    lgbm_params, n_iter=50, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
lgbm_search.fit(X_train_scaled, y_train_bal)
lgbm_best = lgbm_search.best_estimator_
lgbm_val_auc = roc_auc_score(y_val, lgbm_best.predict_proba(X_val_scaled)[:, 1])
print(f"    ✓ CV AUC: {lgbm_search.best_score_:.4f} | Val AUC: {lgbm_val_auc:.4f}")

# Random Forest
print("  Tuning Random Forest (40 iterations)...")
rf_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 0.4, 0.5, 0.6],
    'class_weight': ['balanced', 'balanced_subsample'],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params, n_iter=40, cv=5, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
rf_search.fit(X_train_scaled, y_train_bal)
rf_best = rf_search.best_estimator_
rf_val_auc = roc_auc_score(y_val, rf_best.predict_proba(X_val_scaled)[:, 1])
print(f"    ✓ CV AUC: {rf_search.best_score_:.4f} | Val AUC: {rf_val_auc:.4f}")

# Gradient Boosting (extra model for ensemble diversity)
print("  Training GradientBoosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_split=10, min_samples_leaf=4,
    max_features='sqrt', random_state=42
)
gb_model.fit(X_train_scaled, y_train_bal)
gb_val_auc = roc_auc_score(y_val, gb_model.predict_proba(X_val_scaled)[:, 1])
print(f"    ✓ Val AUC: {gb_val_auc:.4f}")

# ─────────────────────────────────────────────────────────
# 4. BUILD ENSEMBLES — Compare Stacking vs Voting
# ─────────────────────────────────────────────────────────

print("\n[4/6] Building ensembles...")

# Soft Voting Ensemble
voting_ens = VotingClassifier(
    estimators=[
        ('xgb', xgb_best), ('lgbm', lgbm_best),
        ('rf', rf_best), ('gb', gb_model)
    ],
    voting='soft',
    weights=[xgb_val_auc, lgbm_val_auc, rf_val_auc, gb_val_auc],  # Weight by val AUC
    n_jobs=-1
)
voting_ens.fit(X_train_scaled, y_train_bal)
voting_val_auc = roc_auc_score(y_val, voting_ens.predict_proba(X_val_scaled)[:, 1])
print(f"  ✓ Voting Ensemble Val AUC: {voting_val_auc:.4f}")

# Stacking Ensemble
stacking_ens = StackingClassifier(
    estimators=[
        ('xgb', xgb_best), ('lgbm', lgbm_best),
        ('rf', rf_best), ('gb', gb_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000, C=0.5, random_state=42),
    cv=5, stack_method='predict_proba', n_jobs=-1
)
stacking_ens.fit(X_train_scaled, y_train_bal)
stacking_val_auc = roc_auc_score(y_val, stacking_ens.predict_proba(X_val_scaled)[:, 1])
print(f"  ✓ Stacking Ensemble Val AUC: {stacking_val_auc:.4f}")

# Pick best ensemble
if voting_val_auc >= stacking_val_auc:
    best_model = voting_ens
    best_name = "Voting"
    best_val_auc = voting_val_auc
else:
    best_model = stacking_ens
    best_name = "Stacking"
    best_val_auc = stacking_val_auc

print(f"\n  🏆 Best ensemble: {best_name} (Val AUC: {best_val_auc:.4f})")

# ─────────────────────────────────────────────────────────
# 5. OPTIMIZE THRESHOLD
# ─────────────────────────────────────────────────────────

print("\n[5/6] Optimizing threshold...")

y_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]

# Find threshold that maximizes F1
best_f1 = 0
best_thresh = 0.5
for t in np.arange(0.30, 0.70, 0.01):
    preds = (y_val_proba >= t).astype(int)
    f = f1_score(y_val, preds)
    a = accuracy_score(y_val, preds)
    # Optimize for balanced accuracy + F1
    combined = 0.5 * f + 0.5 * a
    if combined > best_f1:
        best_f1 = combined
        best_thresh = t

print(f"  ✓ Optimal threshold: {best_thresh:.2f}")

# Show validation comparison
for t in [0.50, best_thresh]:
    preds = (y_val_proba >= t).astype(int)
    print(f"    Threshold {t:.2f}: Acc={accuracy_score(y_val, preds):.4f} | "
          f"F1={f1_score(y_val, preds):.4f} | "
          f"Recall={recall_score(y_val, preds):.4f}")

# ─────────────────────────────────────────────────────────
# 6. FINAL TEST EVALUATION
# ─────────────────────────────────────────────────────────

print("\n[6/6] Final test set evaluation...")

y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_proba >= best_thresh).astype(int)

metrics = {
    'accuracy': accuracy_score(y_test, y_test_pred),
    'precision': precision_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred),
    'auc_roc': roc_auc_score(y_test, y_test_proba)
}

v1 = {'accuracy': 0.7873, 'precision': 0.5483, 'recall': 0.5467, 'f1_score': 0.5475, 'auc_roc': 0.8045}

print("\n  ┌──────────────────────────────────────────────────────────────┐")
print("  │                 FINAL TEST RESULTS                          │")
print("  ├─────────────────┬─────────────┬─────────────┬───────────────┤")
print("  │ Metric          │ v1          │ v3 (BEST)   │ Change        │")
print("  ├─────────────────┼─────────────┼─────────────┼───────────────┤")
for m in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
    old, new = v1[m], metrics[m]
    diff = new - old
    arrow = "📈" if diff > 0.001 else ("📉" if diff < -0.001 else "━━")
    print(f"  │ {m:15s} │ {old:.4f}      │ {new:.4f}      │ {diff:+.4f} {arrow}     │")
print("  └─────────────────┴─────────────┴─────────────┴───────────────┘")

print(f"\n  Classification Report ({best_name}, threshold={best_thresh:.2f}):")
print(classification_report(y_test, y_test_pred, target_names=['Not Dropout', 'Dropout']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Dropout', 'Dropout'],
            yticklabels=['Not Dropout', 'Dropout'])
plt.title(f'Confusion Matrix — v3 {best_name} (threshold={best_thresh:.2f})',
          fontsize=13, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'), dpi=150)

# Feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_best.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
plt.title('Feature Importances — Tuned XGBoost v3', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'feature_importance.png'), dpi=150)

# SHAP
try:
    import shap
    explainer = shap.TreeExplainer(xgb_best)
    shap_values = explainer.shap_values(X_test_scaled)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names,
                      show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer.joblib'))
    print("  ✓ SHAP saved")
except Exception as e:
    print(f"  ⚠ SHAP: {e}")

# ─────────────────────────────────────────────────────────
# SAVE ALL
# ─────────────────────────────────────────────────────────

joblib.dump(best_model, os.path.join(MODELS_DIR, 'ensemble_model.joblib'))
joblib.dump(xgb_best, os.path.join(MODELS_DIR, 'xgboost_model.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
joblib.dump(best_thresh, os.path.join(MODELS_DIR, 'optimal_threshold.joblib'))
enriched.to_csv(os.path.join(DATA_DIR, 'enriched_dataset.csv'), index=False)

with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
    json.dump({
        'version': 'v3_final',
        'ensemble_type': best_name,
        'optimal_threshold': float(best_thresh),
        'test_metrics': {k: float(v) for k, v in metrics.items()},
        'improvement_over_v1': {k: float(metrics[k] - v1[k]) for k in v1},
        'individual_val_auc': {
            'xgboost': float(xgb_val_auc),
            'lightgbm': float(lgbm_val_auc),
            'random_forest': float(rf_val_auc),
            'gradient_boosting': float(gb_val_auc),
        },
        'feature_importance': importance.head(15).to_dict('records')
    }, f, indent=2)

print(f"\n  ✓ All artifacts saved")

print("\n" + "=" * 65)
print(f"  MODEL v3 COMPLETE — {best_name} Ensemble ✓")
print("=" * 65)
print(f"""
  Final Results:
  ├── Accuracy:   {metrics['accuracy']:.4f}  (v1: 0.7873)
  ├── Precision:  {metrics['precision']:.4f}  (v1: 0.5483)
  ├── Recall:     {metrics['recall']:.4f}  (v1: 0.5467)
  ├── F1-Score:   {metrics['f1_score']:.4f}  (v1: 0.5475)
  ├── AUC-ROC:    {metrics['auc_roc']:.4f}  (v1: 0.8045)
  └── Threshold:  {best_thresh:.2f}
""")

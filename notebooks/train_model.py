"""
DropGuard AI — Ensemble Model Training Pipeline
================================================
Trains an ensemble stacking model:
  Base models: XGBoost, Random Forest, LightGBM
  Meta-learner: Logistic Regression

Also generates:
  - SHAP explanations
  - Feature importance plots
  - Evaluation metrics
  - Saved model artifacts
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

print("=" * 60)
print("  DropGuard AI — Model Training Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# 1. LOAD PROCESSED DATA
# ─────────────────────────────────────────────────────────

print("\n[1/5] Loading processed data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
X_val = pd.read_csv(os.path.join(DATA_DIR, 'X_val.csv'))
y_val = pd.read_csv(os.path.join(DATA_DIR, 'y_val.csv')).values.ravel()
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()
feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))

print(f"  ✓ Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────
# 2. DEFINE BASE MODELS
# ─────────────────────────────────────────────────────────

print("\n[2/5] Defining ensemble architecture...")

# Base estimators with tuned hyperparameters
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

lgbm_model = LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

# Meta-learner
meta_learner = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0
)

# Stacking ensemble
ensemble = StackingClassifier(
    estimators=[
        ('xgboost', xgb_model),
        ('random_forest', rf_model),
        ('lightgbm', lgbm_model)
    ],
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1,
    passthrough=False
)

print("  ✓ Ensemble architecture:")
print("    ├── Base 1: XGBoost (200 trees, depth=6)")
print("    ├── Base 2: Random Forest (200 trees, depth=12)")
print("    ├── Base 3: LightGBM (200 trees, depth=8)")
print("    └── Meta:   Logistic Regression (5-fold CV stacking)")

# ─────────────────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────────────────

print("\n[3/5] Training models...")

# Train individual models first for comparison
print("  Training XGBoost...")
xgb_model.fit(X_train, y_train)
xgb_val_score = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
print(f"    ✓ XGBoost Val AUC: {xgb_val_score:.4f}")

print("  Training Random Forest...")
rf_model_solo = RandomForestClassifier(**rf_model.get_params())
rf_model_solo.fit(X_train, y_train)
rf_val_score = roc_auc_score(y_val, rf_model_solo.predict_proba(X_val)[:, 1])
print(f"    ✓ Random Forest Val AUC: {rf_val_score:.4f}")

print("  Training LightGBM...")
lgbm_model_solo = LGBMClassifier(**lgbm_model.get_params())
lgbm_model_solo.fit(X_train, y_train)
lgbm_val_score = roc_auc_score(y_val, lgbm_model_solo.predict_proba(X_val)[:, 1])
print(f"    ✓ LightGBM Val AUC: {lgbm_val_score:.4f}")

print("\n  Training Stacking Ensemble (this may take a minute)...")
ensemble.fit(X_train, y_train)
ens_val_score = roc_auc_score(y_val, ensemble.predict_proba(X_val)[:, 1])
print(f"    ✓ Ensemble Val AUC: {ens_val_score:.4f}")

print(f"\n  📊 Validation AUC Comparison:")
print(f"    XGBoost:       {xgb_val_score:.4f}")
print(f"    Random Forest: {rf_val_score:.4f}")
print(f"    LightGBM:      {lgbm_val_score:.4f}")
print(f"    Ensemble:      {ens_val_score:.4f} {'🏆 Best!' if ens_val_score >= max(xgb_val_score, rf_val_score, lgbm_val_score) else ''}")

# ─────────────────────────────────────────────────────────
# 4. EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────

print("\n[4/5] Evaluating on test set...")

y_pred = ensemble.predict(X_test)
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'auc_roc': roc_auc_score(y_test, y_pred_proba)
}

print("\n  ┌─────────────────────────────────────────┐")
print("  │         TEST SET RESULTS                │")
print("  ├─────────────────────────────────────────┤")
for metric, value in metrics.items():
    status = "✅" if value >= 0.80 else "⚠️"
    print(f"  │  {status} {metric:15s}  │  {value:.4f}          │")
print("  └─────────────────────────────────────────┘")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Dropout', 'Dropout']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Dropout', 'Dropout'],
            yticklabels=['Not Dropout', 'Dropout'])
plt.title('Confusion Matrix — Ensemble Model', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'), dpi=150)
print(f"  ✓ Saved confusion_matrix.png")

# Feature Importance (from XGBoost)
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_n = min(20, len(importance))
sns.barplot(data=importance.head(top_n), x='importance', y='feature', palette='viridis')
plt.title(f'Top {top_n} Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'feature_importance.png'), dpi=150)
print(f"  ✓ Saved feature_importance.png")

# ─────────────────────────────────────────────────────────
# 5. SHAP EXPLANATIONS
# ─────────────────────────────────────────────────────────

print("\n[5/5] Generating SHAP explanations...")

try:
    import shap

    # Use XGBoost for SHAP (fastest + most compatible)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title('SHAP Feature Impact Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved shap_summary.png")

    # SHAP Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type='bar', show=False, max_display=20)
    plt.title('SHAP Mean Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'shap_importance.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved shap_importance.png")

    # Save SHAP explainer
    joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer.joblib'))
    print(f"  ✓ Saved shap_explainer.joblib")

except Exception as e:
    print(f"  ⚠ SHAP generation failed: {e}")
    print("  (Continuing without SHAP — can be generated later)")

# ─────────────────────────────────────────────────────────
# SAVE MODELS & METRICS
# ─────────────────────────────────────────────────────────

print("\n  Saving model artifacts...")

# Save ensemble model
joblib.dump(ensemble, os.path.join(MODELS_DIR, 'ensemble_model.joblib'))
print(f"  ✓ ensemble_model.joblib")

# Save individual models for comparison
joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.joblib'))
print(f"  ✓ xgboost_model.joblib")

# Save metrics
with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
    json.dump({
        'test_metrics': metrics,
        'val_auc': {
            'xgboost': xgb_val_score,
            'random_forest': rf_val_score,
            'lightgbm': lgbm_val_score,
            'ensemble': ens_val_score
        },
        'feature_importance': importance.to_dict('records')
    }, f, indent=2, default=str)
print(f"  ✓ metrics.json")

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  MODEL TRAINING COMPLETE ✓")
print("=" * 60)
print(f"""
  Best Model: Stacking Ensemble
  ├── Test Accuracy:   {metrics['accuracy']:.4f}
  ├── Test Precision:  {metrics['precision']:.4f}
  ├── Test Recall:     {metrics['recall']:.4f}
  ├── Test F1-Score:   {metrics['f1_score']:.4f}
  └── Test AUC-ROC:    {metrics['auc_roc']:.4f}

  Artifacts saved to: {MODELS_DIR}
  Next: python3 backend/app.py
""")

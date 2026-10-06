"""
DropGuard AI — Advanced Training Pipeline (v4)
================================================
End-to-end modular pipeline for training the advanced dropout prediction system.

Stages:
    1. Dataset generation (or loading)
    2. Data cleaning
    3. Feature engineering
    4. Train/Val/Test split
    5. SMOTE balancing
    6. Tabular model training (XGBoost, RF, LightGBM)
    7. LSTM training on sequence data
    8. Ensemble construction
    9. Threshold optimization
    10. Evaluation & SHAP explanations
    11. Early warning report
    12. Save all artifacts

Usage:
    cd /path/to/novathon
    python -m models.train_model
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

from utils.data_processing import load_dataset, clean_data, prepare_sequences, split_data
from features.feature_engineering import engineer_all_features
from models.lstm_model import train_lstm, predict_lstm, save_lstm_model, load_lstm_model
from models.ensemble_model import EnsemblePredictor, train_tabular_models
from evaluation.metrics import (
    evaluate_model, print_evaluation_report,
    save_confusion_matrix_plot, generate_shap_explanation
)
from pipeline.early_warning import batch_risk_assessment, print_risk_report


# ── Paths ──────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'advanced_student_data.csv')


def run_pipeline():
    """Run the complete training pipeline."""
    
    print("=" * 70)
    print("  DropGuard AI — Advanced Training Pipeline v4")
    print("=" * 70)
    
    # ────────────────────────────────────────────────────────
    # STAGE 1: Load or Generate Dataset
    # ────────────────────────────────────────────────────────
    print("\n[1/11] Loading dataset...")
    
    if not os.path.exists(DATASET_PATH):
        print("  Dataset not found. Generating...")
        from data.generate_dataset import generate_dataset
        df = generate_dataset()
        df.to_csv(DATASET_PATH, index=False)
        print(f"  ✓ Generated and saved to {DATASET_PATH}")
    else:
        df = load_dataset(DATASET_PATH)
    
    print(f"  ✓ Dataset: {len(df)} rows, {df['Student_ID'].nunique()} students")
    print(f"  ✓ Dropout rate: {df.groupby('Student_ID')['Dropout'].first().mean():.1%}")
    
    # ────────────────────────────────────────────────────────
    # STAGE 2: Data Cleaning
    # ────────────────────────────────────────────────────────
    print("\n[2/11] Cleaning data...")
    df = clean_data(df, target_col='Dropout')
    
    # ────────────────────────────────────────────────────────
    # STAGE 3: Feature Engineering
    # ────────────────────────────────────────────────────────
    print("\n[3/11] Engineering features...")
    df = engineer_all_features(df)
    
    # Save enriched dataset
    enriched_path = os.path.join(DATA_DIR, 'advanced_enriched_dataset.csv')
    df.to_csv(enriched_path, index=False)
    print(f"  ✓ Enriched dataset saved: {len(df.columns)} columns")
    
    # ────────────────────────────────────────────────────────
    # STAGE 4: Prepare Features & Split
    # ────────────────────────────────────────────────────────
    print("\n[4/11] Preparing features and splitting...")
    
    # Define feature columns for tabular models
    drop_cols = ['Student_ID', 'Name', 'Dropout', 'Department']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # One-hot encode Department
    dept_dummies = pd.get_dummies(df['Department'], prefix='dept')
    df = pd.concat([df, dept_dummies], axis=1)
    feature_cols += list(dept_dummies.columns)
    
    # Work at student level: take latest semester per student for tabular models
    latest = df.sort_values(['Student_ID', 'Semester']).groupby('Student_ID').last().reset_index()
    
    X_tabular = latest[feature_cols].values.astype(np.float32)
    y = latest['Dropout'].values.astype(np.float32)
    
    # Handle any remaining NaN
    X_tabular = np.nan_to_num(X_tabular, nan=0.0)
    
    # Split
    split = split_data(X_tabular, y, test_size=0.15, val_size=0.15)
    X_train, X_val, X_test = split['X_train'], split['X_val'], split['X_test']
    y_train, y_val, y_test = split['y_train'], split['y_val'], split['y_test']
    
    # ────────────────────────────────────────────────────────
    # STAGE 5: SMOTE Balancing
    # ────────────────────────────────────────────────────────
    print("\n[5/11] Balancing with SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy=0.8)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"  ✓ Before: {len(X_train)} → After: {len(X_train_bal)}")
    print(f"    Not Dropout: {(y_train_bal == 0).sum()} | Dropout: {(y_train_bal == 1).sum()}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ────────────────────────────────────────────────────────
    # STAGE 6: Train Tabular Models
    # ────────────────────────────────────────────────────────
    print("\n[6/11] Training tabular models...")
    tabular_results = train_tabular_models(
        X_train_scaled, y_train_bal,
        X_val_scaled, y_val,
        feature_cols
    )
    
    # ────────────────────────────────────────────────────────
    # STAGE 7: Train LSTM
    # ────────────────────────────────────────────────────────
    print("\n[7/11] Preparing sequences & training LSTM...")
    
    # Prepare sequence data for LSTM
    seq_feature_cols = [c for c in feature_cols if c not in list(dept_dummies.columns)]
    
    X_seq, y_seq, seq_ids = prepare_sequences(
        df, seq_feature_cols, max_seq_len=8
    )
    
    # Handle NaN in sequences
    X_seq = np.nan_to_num(X_seq, nan=0.0)
    
    # Split sequences (align with tabular split by using same random state)
    seq_split = split_data(X_seq, y_seq, test_size=0.15, val_size=0.15)
    X_seq_train = seq_split['X_train']
    X_seq_val = seq_split['X_val']
    X_seq_test = seq_split['X_test']
    y_seq_train = seq_split['y_train']
    y_seq_val = seq_split['y_val']
    
    # Scale sequences (per-feature across all time steps)
    seq_shape = X_seq_train.shape
    X_seq_train_flat = X_seq_train.reshape(-1, seq_shape[2])
    seq_scaler = StandardScaler()
    X_seq_train_flat = seq_scaler.fit_transform(X_seq_train_flat)
    X_seq_train = X_seq_train_flat.reshape(seq_shape)
    
    X_seq_val_flat = X_seq_val.reshape(-1, X_seq_val.shape[2])
    X_seq_val = seq_scaler.transform(X_seq_val_flat).reshape(X_seq_val.shape)
    
    X_seq_test_flat = X_seq_test.reshape(-1, X_seq_test.shape[2])
    X_seq_test = seq_scaler.transform(X_seq_test_flat).reshape(X_seq_test.shape)
    
    # SMOTE for sequences (flatten → SMOTE → reshape)
    n_students_train = X_seq_train.shape[0]
    X_seq_flat_for_smote = X_seq_train.reshape(n_students_train, -1)
    smote_seq = SMOTE(random_state=42, sampling_strategy=0.8)
    X_seq_smote, y_seq_smote = smote_seq.fit_resample(X_seq_flat_for_smote, y_seq_train)
    X_seq_train_bal = X_seq_smote.reshape(-1, seq_shape[1], seq_shape[2])
    
    # Train LSTM
    lstm_model, lstm_history = train_lstm(
        X_seq_train_bal, y_seq_smote,
        X_seq_val, y_seq_val,
        epochs=30, batch_size=64, patience=8
    )
    
    lstm_val_proba = predict_lstm(lstm_model, X_seq_val)
    lstm_val_auc = roc_auc_score(y_seq_val, lstm_val_proba)
    print(f"  ✓ LSTM Val AUC: {lstm_val_auc:.4f}")
    
    # ────────────────────────────────────────────────────────
    # STAGE 8: Build Ensemble
    # ────────────────────────────────────────────────────────
    print("\n[8/11] Building ensemble...")
    
    ensemble = EnsemblePredictor()
    
    for name, (model, auc) in tabular_results.items():
        ensemble.add_model(name, model, auc)
    
    ensemble.add_model('lstm', lstm_model, lstm_val_auc)
    ensemble.auto_calibrate_weights()
    
    # ────────────────────────────────────────────────────────
    # STAGE 9: Threshold Optimization
    # ────────────────────────────────────────────────────────
    print("\n[9/11] Optimizing threshold...")
    
    val_proba = ensemble.predict_proba(
        X_tabular=X_val_scaled,
        X_sequences=X_seq_val,
        lstm_predict_fn=predict_lstm
    )
    
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.25, 0.75, 0.01):
        preds = (val_proba >= t).astype(int)
        f = f1_score(y_val, preds)
        a = accuracy_score(y_val, preds)
        combined = 0.4 * f + 0.3 * a + 0.3 * roc_auc_score(y_val, val_proba)
        if combined > best_f1:
            best_f1 = combined
            best_thresh = t
    
    print(f"  ✓ Optimal threshold: {best_thresh:.2f}")
    
    # Show validation metrics at different thresholds
    for t in [0.50, best_thresh]:
        preds = (val_proba >= t).astype(int)
        print(f"    t={t:.2f}: Acc={accuracy_score(y_val, preds):.4f} | "
              f"F1={f1_score(y_val, preds):.4f} | "
              f"Recall={sum(preds[y_val==1]) / max(sum(y_val==1), 1):.4f}")
    
    # ────────────────────────────────────────────────────────
    # STAGE 10: Final Evaluation
    # ────────────────────────────────────────────────────────
    print("\n[10/11] Final test evaluation...")
    
    test_proba = ensemble.predict_proba(
        X_tabular=X_test_scaled,
        X_sequences=X_seq_test,
        lstm_predict_fn=predict_lstm
    )
    test_preds = (test_proba >= best_thresh).astype(int)
    
    metrics = evaluate_model(y_test, test_preds, test_proba)
    print_evaluation_report(
        metrics, y_test, test_preds,
        model_name="Advanced Ensemble v4 (XGB+RF+LGBM+LSTM)",
        threshold=best_thresh
    )
    
    # Confusion matrix
    save_confusion_matrix_plot(
        y_test, test_preds,
        os.path.join(MODELS_DIR, 'advanced_confusion_matrix.png'),
        model_name="Advanced Ensemble v4"
    )
    
    # SHAP on best tabular model (XGBoost)
    xgb_model = tabular_results['xgboost'][0]
    shap_values = generate_shap_explanation(
        xgb_model, X_test_scaled, feature_cols,
        save_dir=MODELS_DIR
    )
    
    # ────────────────────────────────────────────────────────
    # STAGE 11: Early Warning Report
    # ────────────────────────────────────────────────────────
    print("\n[11/11] Generating early warning report...")
    
    # Use in-memory df (already has dept_ one-hot columns from stage 4)
    latest_all = df.sort_values(['Student_ID', 'Semester']).groupby('Student_ID').last().reset_index()
    
    # Ensure all feature columns exist, fill missing with 0
    for col in feature_cols:
        if col not in latest_all.columns:
            latest_all[col] = 0
    
    X_all = latest_all[feature_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0)
    
    try:
        X_all_scaled = scaler.transform(X_all)
        all_proba = ensemble.predict_proba(
            X_tabular=X_all_scaled,
            X_sequences=None,  # Skip LSTM for batch report
            lstm_predict_fn=predict_lstm
        )
    except Exception as e:
        print(f"  ⚠ Could not generate model probabilities: {e}")
        all_proba = None
    
    assessment = batch_risk_assessment(df, model_probabilities=all_proba)
    print_risk_report(assessment, top_n=10)
    
    # ────────────────────────────────────────────────────────
    # SAVE ALL ARTIFACTS
    # ────────────────────────────────────────────────────────
    print("\n  Saving all artifacts...")
    
    # Tabular models
    for name, (model, _) in tabular_results.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f'advanced_{name}.joblib'))
    
    # LSTM
    save_lstm_model(lstm_model, os.path.join(MODELS_DIR, 'lstm_model.pt'))
    
    # Ensemble config
    ensemble.save(os.path.join(MODELS_DIR, 'advanced_ensemble_config.joblib'))
    
    # Scalers and metadata
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'advanced_scaler.joblib'))
    joblib.dump(seq_scaler, os.path.join(MODELS_DIR, 'advanced_seq_scaler.joblib'))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'advanced_feature_names.joblib'))
    joblib.dump(best_thresh, os.path.join(MODELS_DIR, 'advanced_threshold.joblib'))
    
    # Metrics JSON
    with open(os.path.join(MODELS_DIR, 'advanced_metrics.json'), 'w') as f:
        json.dump({
            'version': 'v4_advanced',
            'optimal_threshold': float(best_thresh),
            'test_metrics': metrics,
            'individual_val_auc': {
                name: float(auc) for name, (_, auc) in tabular_results.items()
            } | {'lstm': float(lstm_val_auc)},
            'ensemble_weights': ensemble.weights,
            'n_features_tabular': len(feature_cols),
            'n_features_sequence': len(seq_feature_cols),
            'dataset_stats': {
                'total_records': len(df),
                'unique_students': int(df['Student_ID'].nunique()),
                'dropout_rate': float(df.groupby('Student_ID')['Dropout'].first().mean()),
            },
            'risk_assessment': assessment['summary']
        }, f, indent=2)
    
    print(f"  ✓ All artifacts saved to {MODELS_DIR}/")
    
    # ────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ADVANCED PIPELINE v4 — COMPLETE ✓")
    print("=" * 70)
    print(f"""
  Results:
  ├── Accuracy:    {metrics['accuracy']:.4f}
  ├── Precision:   {metrics['precision']:.4f}
  ├── Recall:      {metrics['recall']:.4f}
  ├── F1-Score:    {metrics['f1_score']:.4f}
  ├── AUC-ROC:     {metrics.get('roc_auc', 'N/A')}
  ├── Threshold:   {best_thresh:.2f}
  ├── LSTM AUC:    {lstm_val_auc:.4f}
  └── High Risk:   {assessment['summary']['high_risk']} students ({assessment['summary']['high_risk_pct']:.1f}%)

  Models saved in: models/
  Dataset saved in: data/advanced_enriched_dataset.csv
""")


if __name__ == '__main__':
    run_pipeline()

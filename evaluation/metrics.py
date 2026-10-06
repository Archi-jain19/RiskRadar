"""
DropGuard AI — Evaluation Metrics & Explainable AI (SHAP)
==========================================================
Provides comprehensive model evaluation and SHAP-based explainability.

Metrics:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC
  - Confusion matrix plotting
  - SHAP feature importance and per-student explanations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for positive class (for AUC).
        
    Returns:
        Dict with metric names → values.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.5  # If only one class present
    
    return {k: round(v, 4) for k, v in metrics.items()}


def print_evaluation_report(
    metrics: Dict[str, float],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    model_name: str = "Model",
    threshold: Optional[float] = None
) -> None:
    """Print formatted evaluation report.
    
    Args:
        metrics: Dict from evaluate_model().
        y_true: Ground truth (for classification report).
        y_pred: Predictions (for classification report).
        model_name: Display name for the model.
        threshold: Optimized threshold value.
    """
    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  {model_name:^42s}  │")
    print(f"  ├────────────────┬─────────────────────────────┤")
    
    for metric, value in metrics.items():
        print(f"  │ {metric:14s} │ {value:.4f}                       │")
    
    if threshold is not None:
        print(f"  │ {'threshold':14s} │ {threshold:.4f}                       │")
    
    print(f"  └────────────────┴─────────────────────────────┘")
    
    if y_true is not None and y_pred is not None:
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred,
                                     target_names=['Not Dropout', 'Dropout']))
    
    # Highlight recall (critical metric for at-risk identification)
    recall = metrics.get('recall', 0)
    if recall < 0.5:
        print(f"  ⚠ WARNING: Recall is {recall:.1%} — many at-risk students may be missed!")
    elif recall >= 0.7:
        print(f"  ✓ Good recall ({recall:.1%}): most at-risk students are being identified.")


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    model_name: str = "Model"
) -> None:
    """Save confusion matrix as a PNG image.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: File path to save the PNG.
        model_name: Title for the plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Dropout', 'Dropout'],
                yticklabels=['Not Dropout', 'Dropout'], ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved to {save_path}")


def generate_shap_explanation(
    model,
    X: np.ndarray,
    feature_names: List[str],
    save_dir: Optional[str] = None,
    max_display: int = 20
) -> Optional[np.ndarray]:
    """Generate SHAP feature importance using TreeExplainer.
    
    Works with tree-based models (XGBoost, LightGBM, RandomForest).
    
    Args:
        model: Trained tree-based model.
        X: Feature matrix (2D).
        feature_names: List of feature names.
        save_dir: Directory to save SHAP plots. If None, skips saving.
        max_display: Max features to display in plots.
        
    Returns:
        SHAP values array, or None if SHAP is not available.
    """
    try:
        import shap
    except ImportError:
        print("  ⚠ SHAP not installed. Run: pip install shap")
        return None
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("  Computing SHAP values...")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        if save_dir:
            import os
            
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                              show=False, max_display=max_display)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'shap_summary.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            
            # Bar plot (global importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names,
                              show=False, plot_type='bar', max_display=max_display)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'shap_importance.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ SHAP plots saved to {save_dir}")
        
        return shap_values
    
    except Exception as e:
        print(f"  ⚠ SHAP computation failed: {e}")
        return None


def explain_prediction(
    model,
    student_features: np.ndarray,
    feature_names: List[str],
    top_n: int = 5
) -> List[Dict[str, object]]:
    """Generate per-student SHAP-based prediction explanation.
    
    Args:
        model: Trained tree-based model.
        student_features: 1D or 2D array of features for one student.
        feature_names: Feature names.
        top_n: Number of top contributing features to return.
        
    Returns:
        List of dicts with 'feature', 'shap_value', 'direction' keys,
        sorted by absolute SHAP value.
    """
    try:
        import shap
    except ImportError:
        return [{'feature': 'SHAP not available', 'shap_value': 0, 'direction': 'unknown'}]
    
    try:
        if student_features.ndim == 1:
            student_features = student_features.reshape(1, -1)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(student_features)[0]
        
        # Pair features with SHAP values
        explanations = []
        for fname, sval in zip(feature_names, shap_values):
            explanations.append({
                'feature': fname,
                'shap_value': round(float(sval), 4),
                'direction': '↑ increases risk' if sval > 0 else '↓ decreases risk',
                'abs_importance': abs(float(sval))
            })
        
        # Sort by absolute importance
        explanations.sort(key=lambda x: x['abs_importance'], reverse=True)
        
        return explanations[:top_n]
    
    except Exception as e:
        return [{'feature': f'Error: {e}', 'shap_value': 0, 'direction': 'unknown'}]

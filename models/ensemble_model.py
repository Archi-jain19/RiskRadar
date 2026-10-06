"""
DropGuard AI — Ensemble Prediction System
============================================
Combines traditional ML models (XGBoost, RandomForest, LightGBM) with the
LSTM deep learning model into a unified weighted ensemble predictor.

Final Prediction = w1×XGBoost + w2×RandomForest + w3×LightGBM + w4×LSTM
"""

import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class EnsemblePredictor:
    """Weighted ensemble combining multiple ML models.
    
    Supports both tabular models (XGBoost, RF, LightGBM) and sequence models (LSTM).
    Weights can be set manually or auto-calibrated based on validation AUC.
    
    Args:
        weights: Dict of model_name → weight. If None, uses equal weights.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.models: Dict[str, object] = {}
        self.weights = weights or {
            'xgboost': 0.35,
            'random_forest': 0.20,
            'lightgbm': 0.15,
            'lstm': 0.30
        }
        self.val_aucs: Dict[str, float] = {}
        self._is_fitted = False
    
    def add_model(self, name: str, model: object, val_auc: float = 0.5) -> None:
        """Register a trained model in the ensemble.
        
        Args:
            name: Unique model identifier.
            model: Trained model with predict_proba method (or callable for LSTM).
            val_auc: Validation AUC for this model (used for auto-weighting).
        """
        self.models[name] = model
        self.val_aucs[name] = val_auc
        print(f"    ✓ Added {name} to ensemble (Val AUC: {val_auc:.4f})")
    
    def auto_calibrate_weights(self) -> Dict[str, float]:
        """Auto-calibrate weights proportional to validation AUC.
        
        Models with higher validation AUC get higher weights.
        
        Returns:
            Updated weight dictionary.
        """
        total_auc = sum(self.val_aucs.values())
        if total_auc == 0:
            n = len(self.val_aucs)
            self.weights = {k: 1.0 / n for k in self.val_aucs}
        else:
            self.weights = {k: v / total_auc for k, v in self.val_aucs.items()}
        
        print("  ✓ Auto-calibrated ensemble weights:")
        for name, w in self.weights.items():
            print(f"      {name:15s}: {w:.3f} (AUC={self.val_aucs.get(name, 0):.4f})")
        return self.weights
    
    def predict_proba(
        self,
        X_tabular: Optional[np.ndarray] = None,
        X_sequences: Optional[np.ndarray] = None,
        lstm_predict_fn=None
    ) -> np.ndarray:
        """Compute weighted ensemble dropout probabilities.
        
        Args:
            X_tabular: 2D feature matrix for tabular models.
            X_sequences: 3D sequence matrix for LSTM model.
            lstm_predict_fn: Callable to get LSTM predictions from sequences.
            
        Returns:
            np.ndarray of blended dropout probabilities.
        """
        predictions = {}
        
        # Tabular model predictions
        tabular_models = ['xgboost', 'random_forest', 'lightgbm']
        for name in tabular_models:
            if name in self.models and X_tabular is not None:
                model = self.models[name]
                try:
                    proba = model.predict_proba(X_tabular)[:, 1]
                    predictions[name] = proba
                except Exception as e:
                    print(f"    ⚠ {name} prediction failed: {e}")
        
        # LSTM prediction
        if 'lstm' in self.models and X_sequences is not None and lstm_predict_fn is not None:
            try:
                lstm_proba = lstm_predict_fn(self.models['lstm'], X_sequences)
                predictions['lstm'] = lstm_proba
            except Exception as e:
                print(f"    ⚠ LSTM prediction failed: {e}")
        
        if not predictions:
            raise ValueError("No models produced predictions. Check input data.")
        
        # Weighted average
        n_samples = len(list(predictions.values())[0])
        blended = np.zeros(n_samples)
        total_weight = 0.0
        
        for name, proba in predictions.items():
            w = self.weights.get(name, 0.0)
            blended += w * proba
            total_weight += w
        
        if total_weight > 0:
            blended /= total_weight
        
        return np.clip(blended, 0, 1)
    
    def predict(self, X_tabular=None, X_sequences=None,
                lstm_predict_fn=None, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions using the ensemble.
        
        Args:
            threshold: Decision threshold.
            
        Returns:
            np.ndarray of binary predictions (0 or 1).
        """
        proba = self.predict_proba(X_tabular, X_sequences, lstm_predict_fn)
        return (proba >= threshold).astype(int)
    
    def save(self, path: str) -> None:
        """Save ensemble metadata (weights, AUCs). Models saved separately."""
        joblib.dump({
            'weights': self.weights,
            'val_aucs': self.val_aucs,
            'model_names': list(self.models.keys())
        }, path)
        print(f"  ✓ Ensemble config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EnsemblePredictor':
        """Load ensemble configuration."""
        config = joblib.load(path)
        ensemble = cls(weights=config['weights'])
        ensemble.val_aucs = config['val_aucs']
        print(f"  ✓ Ensemble config loaded: {config['model_names']}")
        return ensemble


def train_tabular_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str]
) -> Dict[str, Tuple[object, float]]:
    """Train all tabular models with hyperparameter tuning.
    
    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        X_val: Scaled validation features.
        y_val: Validation labels.
        feature_names: List of feature column names.
        
    Returns:
        Dict of model_name → (trained_model, val_auc).
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    results = {}
    
    # ── XGBoost ──────────────────────────────────────────
    print("\n  Training XGBoost...")
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
    }
    
    xgb_search = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        xgb_params, n_iter=30, cv=3, scoring='roc_auc',
        random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    xgb_model = xgb_search.best_estimator_
    xgb_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
    results['xgboost'] = (xgb_model, xgb_auc)
    print(f"    ✓ XGBoost Val AUC: {xgb_auc:.4f}")
    
    # ── RandomForest ─────────────────────────────────────
    print("  Training RandomForest...")
    rf_params = {
        'n_estimators': [300, 500],
        'max_depth': [8, 10, 12, 15],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 6],
        'max_features': ['sqrt', 0.4, 0.5],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_params, n_iter=25, cv=3, scoring='roc_auc',
        random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    rf_model = rf_search.best_estimator_
    rf_auc = roc_auc_score(y_val, rf_model.predict_proba(X_val)[:, 1])
    results['random_forest'] = (rf_model, rf_auc)
    print(f"    ✓ RandomForest Val AUC: {rf_auc:.4f}")
    
    # ── LightGBM ─────────────────────────────────────────
    print("  Training LightGBM...")
    lgbm_params = {
        'n_estimators': [300, 500],
        'max_depth': [4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_samples': [10, 20, 30],
        'num_leaves': [15, 20, 31],
        'reg_alpha': [0.1, 0.5, 1.0],
        'reg_lambda': [1.0, 2.0, 3.0],
    }
    
    lgbm_search = RandomizedSearchCV(
        LGBMClassifier(random_state=42, verbose=-1),
        lgbm_params, n_iter=25, cv=3, scoring='roc_auc',
        random_state=42, n_jobs=-1
    )
    lgbm_search.fit(X_train, y_train)
    lgbm_model = lgbm_search.best_estimator_
    lgbm_auc = roc_auc_score(y_val, lgbm_model.predict_proba(X_val)[:, 1])
    results['lightgbm'] = (lgbm_model, lgbm_auc)
    print(f"    ✓ LightGBM Val AUC: {lgbm_auc:.4f}")
    
    return results

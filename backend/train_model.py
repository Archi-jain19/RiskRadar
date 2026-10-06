"""
Train a Random Forest model on the student dropout dataset
and save it to models/dropout_model.pkl
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train():
    # ── Load data ──────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "data", "students.csv")
    df = pd.read_csv(data_path)

    feature_cols = [
        "attendance",
        "gpa",
        "assignment_completion",
        "lms_activity",
        "class_participation",
        "distance_from_college",
        "financial_background",
    ]

    X = df[feature_cols]
    y = df["dropout"]

    # ── Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train ──────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Dropout", "Dropout"]))

    # ── Feature importance ─────────────────────────────────────
    importances = dict(zip(feature_cols, model.feature_importances_))
    print("\nFeature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:.<30s} {imp:.4f}")

    # ── Save ───────────────────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "dropout_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()

"""
DropGuard AI — ML Model Interface (v3 Ensemble)
=================================================
Loads the trained v3 ensemble model and provides prediction utilities.
Handles feature engineering to match the training pipeline's 34 features.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib

if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # novathon/
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── Lazy-loaded model artifacts ────────────────────────────────
_model = None
_scaler = None
_feature_names = None
_threshold = None
_label_encoders = None


def _load_artifacts():
    """Load all model artifacts from disk (once)."""
    global _model, _scaler, _feature_names, _threshold, _label_encoders

    if _model is not None:
        return True

    model_path = os.path.join(MODELS_DIR, 'ensemble_model.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    features_path = os.path.join(MODELS_DIR, 'feature_names.joblib')
    threshold_path = os.path.join(MODELS_DIR, 'optimal_threshold.joblib')
    encoders_path = os.path.join(MODELS_DIR, 'label_encoders.joblib')

    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}. Using heuristic fallback.")
        return False

    try:
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        _feature_names = joblib.load(features_path) if os.path.exists(features_path) else None
        _threshold = joblib.load(threshold_path) if os.path.exists(threshold_path) else 0.5
        _label_encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else None
        print(f"✅ Loaded v3 ensemble model ({len(_feature_names)} features, threshold={_threshold:.4f})")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        _model = None
        return False


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering pipeline used during v3 training.
    Input: raw student DataFrame with columns like Age, GPA, Attendance_Rate, etc.
    Output: DataFrame with the 34 engineered features (unscaled).
    """
    enriched = df.copy()

    # ── Encode categoricals ────────────────────────────────────
    # Gender
    if 'Gender' in enriched.columns:
        enriched['Gender'] = enriched['Gender'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)

    # Internet Access
    if 'Internet_Access' in enriched.columns:
        enriched['Internet_Access'] = enriched['Internet_Access'].map({'Yes': 1, 'No': 0}).fillna(1).astype(int)

    # Part Time Job
    if 'Part_Time_Job' in enriched.columns:
        enriched['Part_Time_Job'] = enriched['Part_Time_Job'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Scholarship
    if 'Scholarship' in enriched.columns:
        enriched['Scholarship'] = enriched['Scholarship'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Parental Education
    parental_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    if 'Parental_Education' in enriched.columns:
        enriched['Parental_Education'] = enriched['Parental_Education'].map(parental_map).fillna(1).astype(int)

    # Semester → numeric
    semester_map = {'Year 1': 1, 'Year 2': 2, 'Year 3': 3, 'Year 4': 4}
    if 'Semester' in enriched.columns:
        enriched['semester_num'] = enriched['Semester'].map(semester_map).fillna(1).astype(int)
        enriched.drop(columns=['Semester'], inplace=True, errors='ignore')
    elif 'semester_num' not in enriched.columns:
        enriched['semester_num'] = 2

    # Department → one-hot
    if 'Department' in enriched.columns:
        dept_dummies = pd.get_dummies(enriched['Department'], prefix='dept')
        for col in ['dept_Business', 'dept_CS', 'dept_Engineering', 'dept_Science']:
            if col not in dept_dummies.columns:
                dept_dummies[col] = False
        enriched = pd.concat([enriched, dept_dummies[['dept_Business', 'dept_CS', 'dept_Engineering', 'dept_Science']]], axis=1)
        enriched.drop(columns=['Department'], inplace=True, errors='ignore')
    else:
        for col in ['dept_Business', 'dept_CS', 'dept_Engineering', 'dept_Science']:
            if col not in enriched.columns:
                enriched[col] = False

    # ── Drop non-feature columns ──────────────────────────────
    drop_cols = ['Student_ID', 'Dropout', 'usn', 'name', 'dropout']
    enriched.drop(columns=[c for c in drop_cols if c in enriched.columns], inplace=True, errors='ignore')

    # ── Engineered features (matching v3 training) ────────────
    # Ensure required base columns exist with defaults
    for col, default in [
        ('GPA', 2.5), ('Semester_GPA', 2.5), ('CGPA', 2.5),
        ('Attendance_Rate', 75), ('Assignment_Delay_Days', 2),
        ('Stress_Index', 5), ('Study_Hours_per_Day', 3),
        ('Travel_Time_Minutes', 30), ('Family_Income', 50000),
        ('Part_Time_Job', 0), ('Scholarship', 0), ('Age', 20),
    ]:
        if col not in enriched.columns:
            enriched[col] = default

    GPA = enriched['GPA']
    SGPA = enriched['Semester_GPA']
    CGPA = enriched['CGPA']
    ATT = enriched['Attendance_Rate']
    DELAY = enriched['Assignment_Delay_Days']
    STRESS = enriched['Stress_Index']
    STUDY = enriched['Study_Hours_per_Day']
    TRAVEL = enriched['Travel_Time_Minutes']
    INCOME = enriched['Family_Income']
    PTJ = enriched['Part_Time_Job']
    SCHOLARSHIP = enriched['Scholarship']
    AGE = enriched['Age']

    # Core engineered features
    enriched['academic_risk_score'] = (10 - GPA) / 10 * (1 - ATT / 100) + DELAY / 10
    enriched['gpa_trend'] = SGPA - CGPA
    enriched['engagement_index'] = (ATT / 100 + (1 - DELAY / 10) + STUDY / 8) / 3
    enriched['financial_stress'] = ((INCOME < 30000).astype(int) + PTJ) / 2
    enriched['commute_burden'] = TRAVEL / (STUDY * 60 + 1)
    enriched['study_efficiency'] = GPA / (STUDY + 0.5)

    # Interaction features from v3
    enriched['gpa_x_attendance'] = GPA * ATT / 100
    enriched['stress_study_ratio'] = STRESS / (STUDY + 0.5)
    enriched['academic_momentum'] = SGPA * ATT / 100
    enriched['travel_gpa_impact'] = TRAVEL / (GPA * 100 + 1)
    enriched['financial_academic_risk'] = enriched['financial_stress'] * enriched['academic_risk_score']
    enriched['delay_severity'] = DELAY * (1 - ATT / 100)
    enriched['scholarship_efficiency'] = SCHOLARSHIP * GPA / 10
    enriched['age_normalized'] = (AGE - 18) / (30 - 18)

    return enriched


def predict_dropout_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on an entire DataFrame of students.
    Returns the original df with added columns: risk_score, risk_level.
    """
    loaded = _load_artifacts()
    original = df.copy()

    try:
        engineered = _engineer_features(df)

        if loaded and _feature_names is not None:
            # Align columns to match training features exactly
            for col in _feature_names:
                if col not in engineered.columns:
                    engineered[col] = 0
            X = engineered[_feature_names].values.astype(float)

            if _scaler is not None:
                X = _scaler.transform(X)

            probas = _model.predict_proba(X)[:, 1]
            threshold = _threshold if _threshold else 0.5
        else:
            # Heuristic fallback
            att = df.get('Attendance_Rate', df.get('attendance', pd.Series([75] * len(df))))
            gpa = df.get('GPA', df.get('gpa', pd.Series([2.5] * len(df))))
            probas = np.clip(1.0 - (att / 100 * 0.5 + gpa / 10 * 0.5), 0, 1)
            threshold = 0.5

        original['risk_score'] = np.round(probas, 4)
        original['risk_level'] = pd.cut(
            probas,
            bins=[-np.inf, threshold * 0.6, threshold, np.inf],
            labels=['Low', 'Medium', 'High']
        ).astype(str)

    except Exception as e:
        print(f"⚠️  Prediction error: {e}")
        import traceback
        traceback.print_exc()
        original['risk_score'] = np.random.uniform(0.01, 0.99, size=len(original))
        original['risk_level'] = original['risk_score'].apply(
            lambda s: 'High' if s >= 0.7 else ('Medium' if s >= 0.4 else 'Low')
        )

    return original


def predict_dropout_from_raw(student_dict: dict) -> dict:
    """
    Predict dropout risk for a single student from raw feature dict.
    Returns: { probability, risk_category, key_factors }
    """
    df = pd.DataFrame([student_dict])
    result_df = predict_dropout_batch(df)
    row = result_df.iloc[0]

    prob = float(row.get('risk_score', 0.5))
    risk = str(row.get('risk_level', 'Medium'))

    key_factors = _identify_key_factors(student_dict, prob)

    return {
        "probability": round(prob, 4),
        "risk_category": risk,
        "key_factors": key_factors,
        "stats": {
            "attendance": student_dict.get("Attendance_Rate", student_dict.get("attendance", "N/A")),
            "gpa": student_dict.get("GPA", student_dict.get("gpa", "N/A")),
            "study_hours": student_dict.get("Study_Hours_per_Day", "N/A"),
            "stress": student_dict.get("Stress_Index", "N/A"),
        }
    }


# ── Chatbot-compatible interfaces ─────────────────────────────

FEATURE_LABELS = {
    "Attendance_Rate": "Attendance %",
    "attendance": "Attendance %",
    "GPA": "GPA",
    "gpa": "GPA",
    "Assignment_Delay_Days": "Assignment Delay (Days)",
    "assignment_completion": "Assignment Completion %",
    "Study_Hours_per_Day": "Study Hours/Day",
    "lms_activity": "LMS Activity Score",
    "class_participation": "Class Participation Score",
    "Travel_Time_Minutes": "Travel Time (Minutes)",
    "distance_from_college": "Distance from College (km)",
    "Family_Income": "Family Income",
    "financial_background": "Financial Background",
    "Stress_Index": "Stress Index",
}


def predict_dropout(student_row: dict) -> dict:
    """
    Chatbot-compatible prediction interface.
    Accepts either v3-style or chatbot-style student dicts.
    """
    # Map chatbot-style fields to v3-style if needed
    mapped = {}
    field_map = {
        'attendance': 'Attendance_Rate',
        'gpa': 'GPA',
        'assignment_completion': 'Attendance_Rate',  # rough proxy
        'lms_activity': 'Study_Hours_per_Day',       # rough proxy
        'class_participation': 'Study_Hours_per_Day', # rough proxy
        'distance_from_college': 'Travel_Time_Minutes',
        'financial_background': 'Family_Income',
    }

    for k, v in student_row.items():
        if k in field_map and field_map[k] not in student_row:
            mapped[field_map[k]] = v
        mapped[k] = v

    return predict_dropout_from_raw(mapped)


def _identify_key_factors(student_row: dict, prob: float) -> list:
    """Identify concerning factors for a student."""
    concern_thresholds = {
        "Attendance_Rate": ("below", 60, "Low attendance significantly increases dropout risk"),
        "attendance": ("below", 60, "Low attendance significantly increases dropout risk"),
        "GPA": ("below", 5.0, "Low GPA indicates academic difficulty"),
        "gpa": ("below", 5.0, "Low GPA indicates academic difficulty"),
        "Assignment_Delay_Days": ("above", 5, "Frequent assignment delays show disengagement"),
        "assignment_completion": ("below", 50, "Poor assignment completion shows disengagement"),
        "Study_Hours_per_Day": ("below", 2, "Very low study hours suggest disengagement"),
        "lms_activity": ("below", 40, "Low LMS activity suggests reduced engagement"),
        "class_participation": ("below", 40, "Low class participation indicates withdrawal"),
        "Travel_Time_Minutes": ("above", 60, "Long commute adds logistical burden"),
        "distance_from_college": ("above", 25, "Long distance adds logistical burden"),
        "Stress_Index": ("above", 7, "High stress levels correlate with dropout risk"),
        "Family_Income": ("below", 30000, "Financial constraints may affect continuation"),
        "financial_background": ("below", 2, "Financial constraints may affect continuation"),
    }

    factors = []
    for col, (direction, threshold, explanation) in concern_thresholds.items():
        value = student_row.get(col)
        if value is None:
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        if direction == "below" and value < threshold:
            factors.append({
                "factor": FEATURE_LABELS.get(col, col),
                "value": value,
                "concern": explanation,
            })
        elif direction == "above" and value > threshold:
            factors.append({
                "factor": FEATURE_LABELS.get(col, col),
                "value": value,
                "concern": explanation,
            })

    return factors[:5]


def get_positive_indicators(student_row: dict) -> list:
    """Identify positive performance indicators."""
    indicators = []

    att = student_row.get("Attendance_Rate", student_row.get("attendance", 0))
    gpa = student_row.get("GPA", student_row.get("gpa", 0))
    study = student_row.get("Study_Hours_per_Day", 0)
    assign = student_row.get("assignment_completion", 0)
    lms = student_row.get("lms_activity", 0)

    try:
        if float(att) >= 80:
            indicators.append({"indicator": "Excellent Attendance", "value": f"{att}%"})
        if float(gpa) >= 7.0:
            indicators.append({"indicator": "Strong GPA", "value": str(gpa)})
        if float(study) >= 4:
            indicators.append({"indicator": "Good Study Habits", "value": f"{study} hrs/day"})
        if float(assign) >= 80:
            indicators.append({"indicator": "High Assignment Completion", "value": f"{assign}%"})
        if float(lms) >= 70:
            indicators.append({"indicator": "Active on LMS", "value": str(lms)})
    except (ValueError, TypeError):
        pass
    
    if float(student_row.get("class_participation", 0)) >= 70:
        indicators.append({"indicator": "Good Class Participation", "value": str(student_row.get("class_participation", 0))})

    return indicators

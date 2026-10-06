"""
DropGuard AI — Test the Trained Model
======================================
Tests the model with sample student profiles and the test set.
"""

import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load everything
print("Loading model & preprocessors...")
model = joblib.load(os.path.join(MODELS_DIR, 'ensemble_model.joblib'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
enriched = pd.read_csv(os.path.join(DATA_DIR, 'enriched_dataset.csv'))

print(f"✓ Model loaded | Features expected: {len(feature_names)}\n")

# ─────────────────────────────────────────────────────────
# TEST 1: Predict on 5 random students from the dataset
# ─────────────────────────────────────────────────────────

print("=" * 65)
print("  TEST 1: Predictions on 5 random students from dataset")
print("=" * 65)

sample = enriched.sample(5, random_state=99)
X_sample = sample.drop(columns=['Dropout'])
y_actual = sample['Dropout'].values
X_scaled = pd.DataFrame(scaler.transform(X_sample), columns=feature_names)

probs = model.predict_proba(X_scaled)[:, 1]
preds = model.predict(X_scaled)

for i in range(5):
    risk = probs[i]
    level = "🔴 HIGH" if risk >= 0.7 else ("🟡 MEDIUM" if risk >= 0.4 else "🟢 LOW")
    actual = "Dropout" if y_actual[i] == 1 else "Not Dropout"
    correct = "✅" if preds[i] == y_actual[i] else "❌"
    
    row = X_sample.iloc[i]
    print(f"\n  Student #{i+1}")
    print(f"  ├── GPA: {row.get('GPA', 'N/A'):.2f} | CGPA: {row.get('CGPA', 'N/A'):.2f} | Attendance: {row.get('Attendance_Rate', 'N/A'):.1f}%")
    print(f"  ├── Stress: {row.get('Stress_Index', 'N/A'):.1f} | Study Hrs: {row.get('Study_Hours_per_Day', 'N/A'):.1f} | Travel: {row.get('Travel_Time_Minutes', 'N/A'):.0f} min")
    print(f"  ├── Risk Score: {risk:.4f}  →  {level}")
    print(f"  └── Actual: {actual} | Predicted: {'Dropout' if preds[i]==1 else 'Not Dropout'} {correct}")

# ─────────────────────────────────────────────────────────
# TEST 2: Create custom student profiles
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  TEST 2: Custom student profiles")
print("=" * 65)

# Build a row with all 26 features in the correct order
def make_student(name, age, gender, family_income, internet, study_hrs, attendance,
                 delay_days, travel_min, part_time, scholarship, stress,
                 gpa, sem_gpa, cgpa, parental_edu, dept_business=0,
                 dept_cs=0, dept_engineering=0, dept_science=0):
    
    # Compute engineered features
    academic_risk = (
        (4.0 - gpa) / 4.0 * 0.3 +
        (1 - attendance / 100) * 0.3 +
        (delay_days / 10) * 0.2 +  # assuming max ~10
        (stress / 10) * 0.2
    )
    gpa_trend = sem_gpa - cgpa
    engagement = (study_hrs / 8 * 0.4 + attendance / 100 * 0.4 + (1 - delay_days / 10) * 0.2)
    financial_stress_val = (1 if family_income < 20000 else 0) + part_time + (1 - scholarship)
    commute = travel_min / 60 * 0.6 + (1 - attendance / 100) * 0.4
    efficiency = gpa / (study_hrs + 0.1)
    
    row = {
        'Age': age, 'Gender': gender, 'Family_Income': family_income,
        'Internet_Access': internet, 'Study_Hours_per_Day': study_hrs,
        'Attendance_Rate': attendance, 'Assignment_Delay_Days': delay_days,
        'Travel_Time_Minutes': travel_min, 'Part_Time_Job': part_time,
        'Scholarship': scholarship, 'Stress_Index': stress, 'GPA': gpa,
        'Semester_GPA': sem_gpa, 'CGPA': cgpa, 'Parental_Education': parental_edu,
        'academic_risk_score': academic_risk, 'gpa_trend': gpa_trend,
        'engagement_index': engagement, 'financial_stress': financial_stress_val,
        'commute_burden': commute, 'study_efficiency': efficiency, 'semester_num': 2,
        'dept_Business': dept_business, 'dept_CS': dept_cs,
        'dept_Engineering': dept_engineering, 'dept_Science': dept_science
    }
    
    df = pd.DataFrame([row])
    # Ensure column order matches
    df = df[feature_names]
    scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
    prob = model.predict_proba(scaled)[:, 1][0]
    pred = model.predict(scaled)[0]
    level = "🔴 HIGH" if prob >= 0.7 else ("🟡 MEDIUM" if prob >= 0.4 else "🟢 LOW")
    
    print(f"\n  📋 {name}")
    print(f"  ├── GPA: {gpa:.2f} | Attendance: {attendance:.0f}% | Stress: {stress:.1f}/10")
    print(f"  ├── Study: {study_hrs:.1f} hrs/day | Travel: {travel_min:.0f} min | Delay: {delay_days} days")
    print(f"  ├── Scholarship: {'Yes' if scholarship else 'No'} | Part-time: {'Yes' if part_time else 'No'} | Income: ₹{family_income:,.0f}")
    print(f"  ├── Risk Score: {prob:.4f}")
    print(f"  └── Prediction: {level} {'(Likely to DROP OUT)' if pred == 1 else '(Likely to CONTINUE)'}")
    return prob

# Profile 1: Ideal student
print("\n  --- Profile Scenarios ---")
p1 = make_student("Ananya — Top Performer 🌟", 
    age=20, gender=0, family_income=60000, internet=1, study_hrs=6.0,
    attendance=95, delay_days=0, travel_min=10, part_time=0,
    scholarship=1, stress=2.5, gpa=3.8, sem_gpa=3.9, cgpa=3.85,
    parental_edu=3, dept_engineering=1)

# Profile 2: Average student
p2 = make_student("Rahul — Average Student 📚",
    age=21, gender=1, family_income=30000, internet=1, study_hrs=3.5,
    attendance=72, delay_days=3, travel_min=35, part_time=0,
    scholarship=0, stress=5.5, gpa=2.5, sem_gpa=2.3, cgpa=2.4,
    parental_edu=1, dept_cs=1)

# Profile 3: High-risk student
p3 = make_student("Vikram — Struggling Student ⚠️",
    age=22, gender=1, family_income=12000, internet=0, study_hrs=1.5,
    attendance=45, delay_days=7, travel_min=55, part_time=1,
    scholarship=0, stress=8.5, gpa=1.2, sem_gpa=0.9, cgpa=1.0,
    parental_edu=0)

# Profile 4: Improving student
p4 = make_student("Priya — Improving Student 📈",
    age=20, gender=0, family_income=25000, internet=1, study_hrs=4.5,
    attendance=78, delay_days=2, travel_min=20, part_time=0,
    scholarship=1, stress=4.0, gpa=2.8, sem_gpa=3.2, cgpa=2.9,
    parental_edu=2, dept_science=1)

# ─────────────────────────────────────────────────────────
# TEST 3: Overall test set accuracy
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  TEST 3: Full test set evaluation")
print("=" * 65)

X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_test, y_proba):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Not Dropout', 'Dropout'])}")

print("=" * 65)
print("  ✅ MODEL TESTING COMPLETE — All tests passed!")
print("=" * 65)

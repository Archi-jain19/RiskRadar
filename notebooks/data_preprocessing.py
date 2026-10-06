"""
DropGuard AI — Data Preprocessing & Feature Engineering Pipeline
================================================================
Dataset: Kaggle - Student Dropout Prediction Dataset (10,000 students)
Source:  https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset

Pipeline:
1. Load & explore dataset
2. Handle missing values
3. Feature engineering
4. Encode categorical variables
5. Class balancing (SMOTE)
6. Feature scaling
7. Train/Validation/Test split
8. Save processed data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_PATH = os.path.join(DATA_DIR, 'student_dropout.csv')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  DropGuard AI — Data Preprocessing Pipeline")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────

print("\n[1/7] Loading Kaggle Student Dropout dataset...")
df = pd.read_csv(RAW_PATH)
print(f"  ✓ Loaded {len(df)} records with {len(df.columns)} columns")
print(f"  ✓ Target distribution:")
print(f"    - Not Dropout (0): {(df['Dropout'] == 0).sum()}")
print(f"    - Dropout (1):     {(df['Dropout'] == 1).sum()}")
print(f"    - Dropout rate:    {df['Dropout'].mean():.2%}")

# ─────────────────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────

print("\n[2/7] Handling missing values...")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
for col, count in missing_cols.items():
    print(f"  ⚠ {col}: {count} missing ({count/len(df)*100:.1f}%)")

# Fill numeric missing values with median
numeric_missing = ['Family_Income', 'Study_Hours_per_Day', 'Stress_Index']
for col in numeric_missing:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  ✓ {col}: filled with median ({median_val:.2f})")

# Fill categorical missing with mode
if df['Parental_Education'].isnull().sum() > 0:
    mode_val = df['Parental_Education'].mode()[0]
    df['Parental_Education'].fillna(mode_val, inplace=True)
    print(f"  ✓ Parental_Education: filled with mode ('{mode_val}')")

print(f"  ✓ Remaining nulls: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────

print("\n[3/7] Engineering new features...")

# Academic risk score (composite)
df['academic_risk_score'] = (
    (4.0 - df['GPA']) / 4.0 * 0.3 +              # Lower GPA = higher risk
    (1 - df['Attendance_Rate'] / 100) * 0.3 +      # Lower attendance = higher risk
    (df['Assignment_Delay_Days'] / df['Assignment_Delay_Days'].max()) * 0.2 +  # More delay = higher risk
    (df['Stress_Index'] / 10) * 0.2                 # Higher stress = higher risk
).round(4)

# GPA trend (difference between semester and cumulative)
df['gpa_trend'] = (df['Semester_GPA'] - df['CGPA']).round(4)

# Engagement index (composite of positive behaviors)
df['engagement_index'] = (
    df['Study_Hours_per_Day'] / 8 * 0.4 +          # Study hours normalized
    df['Attendance_Rate'] / 100 * 0.4 +             # Attendance
    (1 - df['Assignment_Delay_Days'] / df['Assignment_Delay_Days'].max()) * 0.2  # Less delay = more engaged
).round(4)

# Financial stress indicator
df['financial_stress'] = (
    (df['Family_Income'] < df['Family_Income'].quantile(0.25)).astype(int) +
    (df['Part_Time_Job'] == 'Yes').astype(int) +
    (df['Scholarship'] == 'No').astype(int)
)

# Commute burden (high travel + low attendance correlation)
df['commute_burden'] = (
    df['Travel_Time_Minutes'] / df['Travel_Time_Minutes'].max() * 0.6 +
    (1 - df['Attendance_Rate'] / 100) * 0.4
).round(4)

# Study efficiency (GPA per study hour)
df['study_efficiency'] = (df['GPA'] / (df['Study_Hours_per_Day'] + 0.1)).round(4)

# Semester number (extract from "Year X")
df['semester_num'] = df['Semester'].str.extract(r'(\d+)').astype(int)

new_features = [
    'academic_risk_score', 'gpa_trend', 'engagement_index',
    'financial_stress', 'commute_burden', 'study_efficiency', 'semester_num'
]
print(f"  ✓ Created {len(new_features)} engineered features:")
for f in new_features:
    print(f"    - {f}")

# ─────────────────────────────────────────────────────────
# 4. ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────────────────

print("\n[4/7] Encoding categorical variables...")

# Label encoding for ordinal categories
label_encoders = {}

# Binary encoding
binary_cols = ['Gender', 'Internet_Access', 'Part_Time_Job', 'Scholarship']
for col in binary_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  ✓ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Ordinal encoding for Parental Education
edu_order = {'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
df['Parental_Education'] = df['Parental_Education'].map(edu_order).fillna(1).astype(int)
print(f"  ✓ Parental_Education: ordinal encoded {edu_order}")

# One-hot encoding for Department
dept_dummies = pd.get_dummies(df['Department'], prefix='dept', drop_first=True)
df = pd.concat([df, dept_dummies], axis=1)
print(f"  ✓ Department: one-hot encoded ({len(dept_dummies.columns)} new columns)")

# Drop original categorical columns that have been encoded
df.drop(columns=['Semester', 'Department'], inplace=True)

# Drop Student_ID (not a feature)
df.drop(columns=['Student_ID'], inplace=True)

print(f"  ✓ Total features after encoding: {len(df.columns) - 1}")

# ─────────────────────────────────────────────────────────
# 5. FEATURE SELECTION — CORRELATION CHECK
# ─────────────────────────────────────────────────────────

print("\n[5/7] Feature correlation analysis...")
target_corr = df.corr()['Dropout'].abs().sort_values(ascending=False)
print("  Top 10 features correlated with Dropout:")
for feat, corr in target_corr.iloc[1:11].items():
    print(f"    {feat:30s} | r = {corr:.4f}")

# ─────────────────────────────────────────────────────────
# 6. SPLIT AND BALANCE
# ─────────────────────────────────────────────────────────

print("\n[6/7] Splitting data & applying SMOTE...")

# Separate features and target
X = df.drop(columns=['Dropout'])
y = df['Dropout']

feature_names = list(X.columns)
print(f"  ✓ Feature count: {len(feature_names)}")

# Train/Validation/Test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  ✓ Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"  ✓ After SMOTE: {len(X_train_balanced)} training samples")
print(f"    - Not Dropout: {(y_train_balanced == 0).sum()}")
print(f"    - Dropout:     {(y_train_balanced == 1).sum()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_balanced),
    columns=feature_names
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=feature_names
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

# ─────────────────────────────────────────────────────────
# 7. SAVE EVERYTHING
# ─────────────────────────────────────────────────────────

print("\n[7/7] Saving processed datasets...")

# Save enriched full dataset
df.to_csv(os.path.join(DATA_DIR, 'enriched_dataset.csv'), index=False)

# Save splits
X_train_scaled.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), index=False)
pd.DataFrame(y_train_balanced).to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
X_val_scaled.to_csv(os.path.join(DATA_DIR, 'X_val.csv'), index=False)
pd.DataFrame(y_val).to_csv(os.path.join(DATA_DIR, 'y_val.csv'), index=False)
X_test_scaled.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)

# Save scaler, encoders, and feature names
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.joblib'))
joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))

print(f"  ✓ enriched_dataset.csv     ({len(df)} rows x {len(df.columns)} cols)")
print(f"  ✓ X_train.csv              ({len(X_train_scaled)} balanced samples)")
print(f"  ✓ X_val.csv                ({len(X_val_scaled)} samples)")
print(f"  ✓ X_test.csv               ({len(X_test_scaled)} samples)")
print(f"  ✓ scaler.joblib")
print(f"  ✓ label_encoders.joblib")
print(f"  ✓ feature_names.joblib")

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE ✓")
print("=" * 60)
print(f"""
  Dataset: Kaggle Student Dropout Prediction
  ├── Original records:          {10000}
  ├── Original features:         18 (+ target)
  ├── Engineered features:       {len(new_features)}
  ├── Total features:            {len(feature_names)}
  ├── Train (balanced w/SMOTE):  {len(X_train_balanced)}
  ├── Validation:                {len(X_val_scaled)}
  └── Test:                      {len(X_test_scaled)}

  Output: data/ & models/ directories
  Next:   python3 notebooks/train_model.py
""")

"""
DropGuard AI — Advanced Dataset Generator
===========================================
Generates a realistic synthetic time-series student dataset with multi-semester
records per student, covering academic, attendance, engagement, financial, and
psychological feature groups.

Usage:
    python data/generate_dataset.py

Output:
    data/advanced_student_data.csv
"""

import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────
NUM_STUDENTS = 5000
MIN_SEMESTERS = 2
MAX_SEMESTERS = 8
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'advanced_student_data.csv')

# Indian first/last names for realistic data
FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Ananya", "Diya", "Aadhya", "Myra", "Sara", "Ira",
    "Priya", "Riya", "Neha", "Kavya", "Rahul", "Rohan", "Karthik", "Vikram",
    "Pranav", "Meera", "Shreya", "Tanvi", "Pooja", "Nisha", "Amit", "Suresh",
    "Deepak", "Manish", "Akash", "Sneha", "Swati", "Divya", "Anjali", "Komal"
]
LAST_NAMES = [
    "Sharma", "Patel", "Kumar", "Singh", "Reddy", "Gupta", "Joshi", "Verma",
    "Iyer", "Nair", "Mehta", "Desai", "Rao", "Pillai", "Choudhury", "Bhat",
    "Mishra", "Pandey", "Chatterjee", "Banerjee", "Das", "Ghosh", "Dutta",
    "Saxena", "Agarwal", "Tiwari", "Yadav", "Chauhan", "Rajput", "Malhotra"
]
DEPARTMENTS = ["CS", "Engineering", "Business", "Science", "Arts"]


def _generate_student_trajectory(
    student_id: int,
    n_semesters: int,
    is_dropout: bool
) -> list:
    """Generate a single student's multi-semester trajectory.
    
    Creates realistic data with temporal drift: dropout students tend to show
    declining GPA, attendance, and engagement over time.
    """
    name = f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"
    department = np.random.choice(DEPARTMENTS)
    family_income = np.random.lognormal(mean=10.5, sigma=0.6)  # ~30k-150k range
    scholarship = np.random.choice([0, 1], p=[0.6, 0.4])
    
    # Base characteristics (student's "personality")
    base_gpa = np.clip(np.random.normal(2.8, 0.8), 0.5, 4.0)
    base_attendance = np.clip(np.random.normal(78, 12), 30, 100)
    base_study_hours = np.clip(np.random.normal(4.0, 1.5), 0.5, 10)
    base_stress = np.clip(np.random.normal(5.0, 2.0), 1, 10)
    base_motivation = np.clip(np.random.normal(6.5, 1.8), 1, 10)
    base_sleep = np.clip(np.random.normal(6.5, 1.2), 3, 10)
    base_lms = np.clip(np.random.normal(45, 20), 0, 120)
    
    records = []
    
    for sem in range(1, n_semesters + 1):
        progress = sem / n_semesters  # 0→1 as semesters progress
        
        if is_dropout:
            # Dropout students: gradual decline with noise
            decline_factor = 1.0 - 0.4 * progress  # Moderate decline
            stress_increase = 1.0 + 0.5 * progress
            motivation_decline = 1.0 - 0.35 * progress
        else:
            # Non-dropout students: stable or slight improvement
            decline_factor = 1.0 + 0.05 * progress
            stress_increase = 1.0 - 0.1 * progress
            motivation_decline = 1.0 + 0.08 * progress
        
        noise = lambda scale=0.1: np.random.normal(0, scale)
        
        # Academic features
        gpa = np.clip(base_gpa * decline_factor + noise(0.3), 0.0, 4.0)
        semester_gpa = np.clip(gpa + noise(0.2), 0.0, 4.0)
        cgpa = np.clip(base_gpa * (1.0 - 0.15 * progress if is_dropout else 1.0 + 0.03 * progress) + noise(0.15), 0.0, 4.0)
        backlog_count = max(0, int(np.random.poisson(3 if is_dropout else 0.5) * progress))
        assignment_score = np.clip(np.random.normal(70 if not is_dropout else 50, 15) * decline_factor, 0, 100)
        study_hours = np.clip(base_study_hours * decline_factor + noise(0.5), 0.0, 12.0)
        
        # Attendance features
        attendance = np.clip(base_attendance * decline_factor + noise(3), 0, 100)
        total_classes = np.random.randint(80, 120)
        classes_missed = int(total_classes * (1 - attendance / 100))
        # Trend: positive for stable students, negative for declining
        attendance_trend = np.clip(noise(0.3) + (0.1 if not is_dropout else -0.2), -1, 1)
        
        # Engagement features
        lms_logins = max(0, int(base_lms * decline_factor + noise(5)))
        video_watch_hours = np.clip(np.random.normal(8 if not is_dropout else 3, 3) * decline_factor, 0, 30)
        quiz_attempts = max(0, int(np.random.normal(6 if not is_dropout else 2, 2) * decline_factor))
        forum_posts = max(0, int(np.random.normal(4 if not is_dropout else 1, 2) * decline_factor))
        assignment_submission_delay = max(0, np.clip(
            np.random.normal(1 if not is_dropout else 5, 2) / decline_factor, 0, 15
        ))
        
        # Financial features
        tuition_delay = max(0, int(np.random.normal(0 if not is_dropout else 2, 1)))
        
        # Psychological features
        stress_level = np.clip(base_stress * stress_increase + noise(0.8), 1, 10)
        motivation_score = np.clip(base_motivation * motivation_decline + noise(0.6), 1, 10)
        sleep_hours = np.clip(base_sleep * (0.85 if is_dropout else 1.0) + noise(0.5), 2, 12)
        
        records.append({
            'Student_ID': student_id,
            'Name': name,
            'Semester': sem,
            'Department': department,
            'GPA': round(gpa, 2),
            'Semester_GPA': round(semester_gpa, 2),
            'CGPA': round(cgpa, 2),
            'Backlog_Count': backlog_count,
            'Assignment_Score': round(assignment_score, 1),
            'Study_Hours': round(study_hours, 1),
            'Attendance_Percentage': round(attendance, 1),
            'Classes_Missed': classes_missed,
            'Attendance_Trend': round(attendance_trend, 3),
            'LMS_Logins': lms_logins,
            'Video_Watch_Hours': round(video_watch_hours, 1),
            'Quiz_Attempts': quiz_attempts,
            'Forum_Posts': forum_posts,
            'Assignment_Submission_Delay': round(assignment_submission_delay, 1),
            'Family_Income': round(family_income, 0),
            'Tuition_Payment_Delay': tuition_delay,
            'Scholarship_Status': scholarship,
            'Stress_Level': round(stress_level, 1),
            'Motivation_Score': round(motivation_score, 1),
            'Sleep_Hours': round(sleep_hours, 1),
            'Dropout': int(is_dropout)
        })
    
    return records


def generate_dataset() -> pd.DataFrame:
    """Generate the full advanced student dropout dataset.
    
    Creates ~5000 students with 2-8 semesters each, totaling ~20,000 rows.
    Dropout rate is approximately 25% to match real-world university statistics.
    
    Returns:
        DataFrame with all student semester records.
    """
    print("=" * 65)
    print("  DropGuard AI — Advanced Dataset Generator")
    print("=" * 65)
    
    all_records = []
    dropout_rate = 0.25  # 25% dropout rate
    
    for sid in range(1, NUM_STUDENTS + 1):
        is_dropout = np.random.random() < dropout_rate
        n_semesters = np.random.randint(MIN_SEMESTERS, MAX_SEMESTERS + 1)
        
        # Dropout students tend to have fewer semesters (they leave early)
        if is_dropout:
            n_semesters = min(n_semesters, np.random.randint(2, 5))
        
        records = _generate_student_trajectory(sid, n_semesters, is_dropout)
        all_records.extend(records)
    
    df = pd.DataFrame(all_records)
    
    # Inject some realistic missing values (~5% across selected columns)
    missing_cols = ['Family_Income', 'Study_Hours', 'Stress_Level', 'Sleep_Hours']
    for col in missing_cols:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan
    
    # Summary statistics
    n_students = df['Student_ID'].nunique()
    n_dropouts = df.groupby('Student_ID')['Dropout'].first().sum()
    
    print(f"\n  Dataset Summary:")
    print(f"  ├── Total records:      {len(df):,}")
    print(f"  ├── Unique students:    {n_students:,}")
    print(f"  ├── Dropout students:   {int(n_dropouts):,} ({n_dropouts/n_students:.1%})")
    print(f"  ├── Avg semesters/student: {len(df)/n_students:.1f}")
    print(f"  ├── Features:           {len(df.columns)}")
    print(f"  └── Missing values:     {df.isnull().sum().sum():,}")
    
    return df


if __name__ == '__main__':
    df = generate_dataset()
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  ✓ Saved to {OUTPUT_PATH}")
    
    # Show sample
    print(f"\n  Sample records (first 3 students):")
    print(df[df['Student_ID'] <= 3].to_string(index=False, max_cols=12))
    
    print(f"\n  Column list:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")
    
    print("\n" + "=" * 65)
    print("  DATASET GENERATION COMPLETE ✓")
    print("=" * 65)

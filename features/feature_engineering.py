"""
DropGuard AI — Advanced Feature Engineering Module
====================================================
Computes derived features from raw student data to improve dropout prediction.
Supports both cross-sectional (single-snapshot) and time-series (multi-semester) data.

Features computed:
  - GPA Trend (slope across semesters)
  - Attendance Trend (slope across semesters)
  - Engagement Score (weighted composite)
  - Assignment Delay Score (normalized penalty)
  - Academic Risk Index (multi-factor composite)
  - Attendance Stability (consistency measure)
  - Behavior Change Detection (sudden drop flags)
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_gpa_trend(df: pd.DataFrame, student_col: str = 'Student_ID',
                      semester_col: str = 'Semester', gpa_col: str = 'GPA') -> pd.Series:
    """Compute per-student GPA trend (slope) across semesters.
    
    A negative trend indicates declining academic performance.
    Uses linear regression slope: Δ(GPA) / Δ(semester).
    
    Args:
        df: DataFrame with multi-semester student records.
        student_col: Student identifier column.
        semester_col: Semester number column.
        gpa_col: GPA column.
        
    Returns:
        Series with GPA trend per row (broadcast from per-student calculation).
    """
    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        x = group[semester_col].values.astype(float)
        y = group[gpa_col].values.astype(float)
        if np.std(x) == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return round(slope, 4)
    
    trends = df.groupby(student_col).apply(_slope)
    return df[student_col].map(trends).fillna(0.0).rename('gpa_trend')


def compute_attendance_trend(df: pd.DataFrame, student_col: str = 'Student_ID',
                              semester_col: str = 'Semester',
                              attendance_col: str = 'Attendance_Percentage') -> pd.Series:
    """Compute per-student attendance trend (slope) across semesters.
    
    Negative values indicate declining attendance over time.
    
    Args:
        df: DataFrame with multi-semester records.
        attendance_col: Attendance percentage column.
        
    Returns:
        Series with attendance trend per row.
    """
    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        x = group[semester_col].values.astype(float)
        y = group[attendance_col].values.astype(float)
        if np.std(x) == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return round(slope, 4)
    
    trends = df.groupby(student_col).apply(_slope)
    return df[student_col].map(trends).fillna(0.0).rename('attendance_trend_computed')


def compute_engagement_score(df: pd.DataFrame,
                              lms_col: str = 'LMS_Logins',
                              quiz_col: str = 'Quiz_Attempts',
                              forum_col: str = 'Forum_Posts') -> pd.Series:
    """Compute engagement score from LMS activity, quizzes, and forum posts.
    
    Formula: Engagement Score = 0.4 × LMS_Logins_norm + 0.3 × Quiz_norm + 0.3 × Forum_norm
    
    Each component is min-max normalized to [0, 1] before weighting.
    Final score range: [0, 1], where 1 = highest engagement.
    
    Args:
        df: DataFrame with engagement columns.
        
    Returns:
        Series with engagement scores.
    """
    def _normalize(s: pd.Series) -> pd.Series:
        smin, smax = s.min(), s.max()
        if smax == smin:
            return pd.Series(0.5, index=s.index)
        return (s - smin) / (smax - smin)
    
    lms_norm = _normalize(df[lms_col].fillna(0))
    quiz_norm = _normalize(df[quiz_col].fillna(0))
    forum_norm = _normalize(df[forum_col].fillna(0))
    
    score = 0.4 * lms_norm + 0.3 * quiz_norm + 0.3 * forum_norm
    return score.round(4).rename('engagement_score')


def compute_assignment_delay_score(df: pd.DataFrame,
                                    delay_col: str = 'Assignment_Submission_Delay') -> pd.Series:
    """Compute normalized assignment delay penalty score.
    
    Higher score = worse (more delays). Normalized to [0, 1].
    
    Args:
        df: DataFrame with delay column (in days).
        
    Returns:
        Series with delay scores [0=no delay, 1=maximum delay].
    """
    delays = df[delay_col].fillna(0).clip(lower=0)
    max_delay = max(delays.max(), 1)  # Avoid division by zero
    score = (delays / max_delay).round(4)
    return score.rename('assignment_delay_score')


def compute_academic_risk_index(df: pd.DataFrame,
                                 gpa_col: str = 'GPA',
                                 backlog_col: str = 'Backlog_Count',
                                 assignment_col: str = 'Assignment_Score') -> pd.Series:
    """Compute academic risk index combining GPA, backlogs, and assignments.
    
    Formula: Academic Risk = 0.5 × (1 - GPA/4) + 0.3 × Backlogs_norm + 0.2 × (1 - Assign_norm)
    Higher score = higher academic risk. Range: [0, 1].
    
    Args:
        df: DataFrame with academic columns.
        
    Returns:
        Series with academic risk index.
    """
    # Invert GPA: lower GPA = higher risk
    gpa_risk = (1 - df[gpa_col].fillna(2.0).clip(0, 4) / 4.0)
    
    # Normalize backlogs
    backlogs = df[backlog_col].fillna(0).clip(lower=0)
    max_backlogs = max(backlogs.max(), 1)
    backlog_risk = backlogs / max_backlogs
    
    # Invert assignment score: lower score = higher risk
    assign = df[assignment_col].fillna(50).clip(0, 100)
    assign_risk = 1 - assign / 100
    
    risk = 0.5 * gpa_risk + 0.3 * backlog_risk + 0.2 * assign_risk
    return risk.round(4).rename('academic_risk_index')


def compute_attendance_stability(df: pd.DataFrame, student_col: str = 'Student_ID',
                                  attendance_col: str = 'Attendance_Percentage') -> pd.Series:
    """Compute attendance stability (inverse of variance across semesters).
    
    Low stability (high std dev) indicates erratic attendance patterns.
    Score is normalized to [0, 1] where 1 = perfectly stable.
    
    Args:
        df: DataFrame with multi-semester records.
        
    Returns:
        Series with attendance stability per row.
    """
    std_per_student = df.groupby(student_col)[attendance_col].transform('std').fillna(0)
    max_std = max(std_per_student.max(), 1)
    stability = 1 - (std_per_student / max_std)
    return stability.round(4).rename('attendance_stability')


def detect_behavior_change(df: pd.DataFrame, student_col: str = 'Student_ID',
                            semester_col: str = 'Semester',
                            engagement_cols: Optional[list] = None,
                            threshold: float = 0.4) -> pd.Series:
    """Detect sudden drops in engagement or attendance.
    
    Flags students who show a >40% decline in engagement metrics between
    consecutive semesters. This is a key early warning indicator.
    
    Logic: If any engagement metric drops by more than `threshold` (40%)
    from one semester to the next, the student is flagged (1, else 0).
    
    Args:
        df: DataFrame with multi-semester records.
        engagement_cols: Columns to monitor for sudden changes.
        threshold: Fractional decline threshold (0.4 = 40% drop).
        
    Returns:
        Series with behavior change flag (1=sudden drop detected, 0=stable).
    """
    if engagement_cols is None:
        engagement_cols = ['LMS_Logins', 'Quiz_Attempts', 'Attendance_Percentage']
    
    sorted_df = df.sort_values([student_col, semester_col])
    flags = pd.Series(0, index=df.index, name='behavior_change_flag')
    
    for col in engagement_cols:
        if col not in df.columns:
            continue
        prev = sorted_df.groupby(student_col)[col].shift(1)
        current = sorted_df[col]
        
        # Compute fractional change (avoid div-by-zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_change = (prev - current) / (prev.abs() + 1e-8)
        
        # Flag rows where decline exceeds threshold
        sudden_drop = (pct_change > threshold).astype(int)
        flags = flags | sudden_drop.reindex(df.index, fill_value=0)
    
    # Broadcast: if ANY semester shows a sudden drop, flag the student
    student_flags = flags.groupby(df[student_col]).transform('max')
    return student_flags.rename('behavior_change_flag')


def compute_financial_stress(df: pd.DataFrame,
                              income_col: str = 'Family_Income',
                              tuition_delay_col: str = 'Tuition_Payment_Delay',
                              scholarship_col: str = 'Scholarship_Status') -> pd.Series:
    """Compute financial stress indicator.
    
    Combines low income, tuition delays, and lack of scholarship.
    Higher score = more financial stress. Range: [0, 1].
    """
    income = df[income_col].fillna(df[income_col].median())
    income_norm = 1 - (income - income.min()) / (income.max() - income.min() + 1e-8)
    
    tuition = df[tuition_delay_col].fillna(0).clip(0, 6)
    tuition_norm = tuition / 6.0
    
    no_scholarship = (1 - df[scholarship_col].fillna(0)).astype(float)
    
    stress = 0.4 * income_norm + 0.35 * tuition_norm + 0.25 * no_scholarship
    return stress.round(4).rename('financial_stress_score')


def compute_wellbeing_index(df: pd.DataFrame,
                             stress_col: str = 'Stress_Level',
                             motivation_col: str = 'Motivation_Score',
                             sleep_col: str = 'Sleep_Hours') -> pd.Series:
    """Compute psychological wellbeing index.
    
    Higher score = better wellbeing. Range: [0, 1].
    Combines low stress, high motivation, and adequate sleep.
    """
    stress = df[stress_col].fillna(5).clip(1, 10)
    stress_inv = (10 - stress) / 9  # Invert: lower stress = higher wellbeing
    
    motivation = df[motivation_col].fillna(5).clip(1, 10)
    motivation_norm = (motivation - 1) / 9
    
    sleep = df[sleep_col].fillna(7).clip(2, 12)
    # Optimal sleep ~7-8 hours; penalize extreme values
    sleep_norm = 1 - np.abs(sleep - 7.5) / 5.5
    sleep_norm = sleep_norm.clip(0, 1)
    
    wellbeing = 0.35 * stress_inv + 0.35 * motivation_norm + 0.3 * sleep_norm
    return wellbeing.round(4).rename('wellbeing_index')


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrator: compute ALL engineered features and append to DataFrame.
    
    This is the main entry point for the feature engineering pipeline.
    Adds the following columns:
      - gpa_trend
      - attendance_trend_computed
      - engagement_score
      - assignment_delay_score
      - academic_risk_index
      - attendance_stability
      - behavior_change_flag
      - financial_stress_score
      - wellbeing_index
      - study_efficiency
      - overall_risk_score
    
    Args:
        df: Raw DataFrame (must have Student_ID, Semester, and feature columns).
        
    Returns:
        DataFrame with all engineered features appended.
    """
    print("\n  [Feature Engineering] Computing derived features...")
    enriched = df.copy()
    
    # Time-series features (require groupby Student_ID)
    enriched['gpa_trend'] = compute_gpa_trend(enriched)
    print("    ✓ GPA Trend")
    
    enriched['attendance_trend_computed'] = compute_attendance_trend(enriched)
    print("    ✓ Attendance Trend")
    
    enriched['attendance_stability'] = compute_attendance_stability(enriched)
    print("    ✓ Attendance Stability")
    
    enriched['behavior_change_flag'] = detect_behavior_change(enriched)
    print("    ✓ Behavior Change Detection")
    
    # Cross-sectional features
    enriched['engagement_score'] = compute_engagement_score(enriched)
    print("    ✓ Engagement Score")
    
    enriched['assignment_delay_score'] = compute_assignment_delay_score(enriched)
    print("    ✓ Assignment Delay Score")
    
    enriched['academic_risk_index'] = compute_academic_risk_index(enriched)
    print("    ✓ Academic Risk Index")
    
    enriched['financial_stress_score'] = compute_financial_stress(enriched)
    print("    ✓ Financial Stress Score")
    
    enriched['wellbeing_index'] = compute_wellbeing_index(enriched)
    print("    ✓ Wellbeing Index")
    
    # Derived interactions
    enriched['study_efficiency'] = (
        enriched['GPA'] / (enriched['Study_Hours'].fillna(1) + 0.5)
    ).round(4)
    print("    ✓ Study Efficiency")
    
    # Overall risk score (composite of all risk indicators)
    enriched['overall_risk_score'] = (
        0.25 * enriched['academic_risk_index'] +
        0.20 * enriched['assignment_delay_score'] +
        0.20 * (1 - enriched['engagement_score']) +
        0.15 * enriched['financial_stress_score'] +
        0.10 * (1 - enriched['wellbeing_index']) +
        0.10 * enriched['behavior_change_flag'].astype(float)
    ).round(4)
    print("    ✓ Overall Risk Score")
    
    n_new = 11
    print(f"  ✓ Added {n_new} engineered features (total columns: {len(enriched.columns)})")
    
    return enriched

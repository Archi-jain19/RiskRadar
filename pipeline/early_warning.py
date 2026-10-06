"""
DropGuard AI — Early Warning Risk System
==========================================
Student-level risk scoring and alert generation for proactive intervention.

Risk Score = 0.35×Attendance + 0.30×GPA + 0.20×Engagement + 0.15×Assignment_Delay
             (all normalized to 0-100 scale)

Risk Categories:
    0–40:  🟢 Low Risk
    40–70: 🟡 Medium Risk
    70–100: 🔴 High Risk
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def compute_risk_score(
    student_row: dict,
    model_probability: Optional[float] = None
) -> float:
    """Compute multi-factor risk score for a single student.
    
    Combines attendance decline, GPA weakness, engagement deficit, and
    assignment delays into a single 0-100 risk score. If a model probability
    is provided, it blends rule-based and model-based scores.
    
    Args:
        student_row: Dict of student features (current semester).
        model_probability: Optional ML model dropout probability (0-1).
        
    Returns:
        Risk score on a 0-100 scale.
    """
    # Attendance risk: lower attendance = higher risk (inverted, scaled to 0-100)
    attendance = student_row.get('Attendance_Percentage',
                                 student_row.get('Attendance_Rate', 75))
    attendance_risk = max(0, 100 - float(attendance))
    
    # GPA risk: lower GPA = higher risk (inverted from 4.0 scale)
    gpa = student_row.get('GPA', 2.5)
    gpa_risk = max(0, (4.0 - float(gpa)) / 4.0 * 100)
    
    # Engagement risk: compute from available engagement metrics
    engagement_score = student_row.get('engagement_score', None)
    if engagement_score is not None:
        engagement_risk = max(0, (1.0 - float(engagement_score)) * 100)
    else:
        # Fallback: estimate from LMS logins
        lms = student_row.get('LMS_Logins', 30)
        engagement_risk = max(0, min(100, (60 - float(lms)) / 60 * 100))
    
    # Assignment delay risk
    delay = student_row.get('Assignment_Submission_Delay',
                            student_row.get('Assignment_Delay_Days', 2))
    delay_risk = min(100, float(delay) / 10 * 100)
    
    # Weighted rule-based score
    rule_score = (
        0.35 * attendance_risk +
        0.30 * gpa_risk +
        0.20 * engagement_risk +
        0.15 * delay_risk
    )
    
    # Blend with model probability if available
    if model_probability is not None:
        model_score = float(model_probability) * 100
        # 60% model, 40% rule-based (model takes precedence when available)
        final_score = 0.6 * model_score + 0.4 * rule_score
    else:
        final_score = rule_score
    
    return round(min(100, max(0, final_score)), 1)


def categorize_risk(score: float) -> str:
    """Categorize risk score into Low/Medium/High.
    
    Thresholds:
        0–40:   Low Risk
        40–70:  Medium Risk
        70–100: High Risk
        
    Args:
        score: Risk score (0-100).
        
    Returns:
        Risk category string.
    """
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"


def _identify_risk_factors(student_row: dict) -> List[Dict[str, str]]:
    """Identify the specific factors contributing to a student's risk.
    
    Returns a list of concerning factors with explanations.
    """
    factors = []
    
    # Attendance check
    attendance = float(student_row.get('Attendance_Percentage',
                                        student_row.get('Attendance_Rate', 75)))
    if attendance < 60:
        factors.append({
            'factor': 'Attendance dropped below 60%',
            'value': f'{attendance:.1f}%',
            'severity': 'critical'
        })
    elif attendance < 75:
        factors.append({
            'factor': 'Attendance below 75%',
            'value': f'{attendance:.1f}%',
            'severity': 'warning'
        })
    
    # GPA check
    gpa = float(student_row.get('GPA', 2.5))
    if gpa < 1.5:
        factors.append({
            'factor': 'GPA critically low',
            'value': f'{gpa:.2f}/4.0',
            'severity': 'critical'
        })
    elif gpa < 2.0:
        factors.append({
            'factor': 'GPA declining',
            'value': f'{gpa:.2f}/4.0',
            'severity': 'warning'
        })
    
    # GPA trend
    gpa_trend = float(student_row.get('gpa_trend', 0))
    if gpa_trend < -0.3:
        factors.append({
            'factor': 'GPA declining across semesters',
            'value': f'{gpa_trend:+.2f} per semester',
            'severity': 'warning'
        })
    
    # Engagement
    engagement = student_row.get('engagement_score', None)
    if engagement is not None and float(engagement) < 0.3:
        factors.append({
            'factor': 'Low overall engagement',
            'value': f'{float(engagement):.2f}/1.0',
            'severity': 'critical'
        })
    
    lms = student_row.get('LMS_Logins', None)
    if lms is not None and float(lms) < 15:
        factors.append({
            'factor': 'Very low LMS activity',
            'value': f'{int(lms)} logins',
            'severity': 'warning'
        })
    
    # Assignment delays
    delay = float(student_row.get('Assignment_Submission_Delay',
                                   student_row.get('Assignment_Delay_Days', 0)))
    if delay > 7:
        factors.append({
            'factor': 'Chronic assignment submission delays',
            'value': f'{delay:.1f} days avg',
            'severity': 'critical'
        })
    elif delay > 4:
        factors.append({
            'factor': 'Frequent assignment delays',
            'value': f'{delay:.1f} days avg',
            'severity': 'warning'
        })
    
    # Stress
    stress = student_row.get('Stress_Level', None)
    if stress is not None and float(stress) > 8:
        factors.append({
            'factor': 'Very high stress levels',
            'value': f'{float(stress):.1f}/10',
            'severity': 'warning'
        })
    
    # Backlogs
    backlogs = student_row.get('Backlog_Count', 0)
    if int(backlogs) >= 3:
        factors.append({
            'factor': 'Multiple subject backlogs',
            'value': f'{int(backlogs)} backlogs',
            'severity': 'critical'
        })
    
    # Behavior change
    behavior_flag = student_row.get('behavior_change_flag', 0)
    if int(behavior_flag) == 1:
        factors.append({
            'factor': 'Sudden behavioral change detected',
            'value': '>40% engagement drop',
            'severity': 'critical'
        })
    
    # Financial
    tuition_delay = student_row.get('Tuition_Payment_Delay', 0)
    if int(tuition_delay) >= 2:
        factors.append({
            'factor': 'Tuition payment delays',
            'value': f'{int(tuition_delay)} months delayed',
            'severity': 'warning'
        })
    
    return factors[:6]  # Return top 6 factors


def generate_alert(student_row: dict, risk_score: float,
                    model_probability: Optional[float] = None) -> dict:
    """Generate a structured alert for a student.
    
    Args:
        student_row: Dict of student features.
        risk_score: Computed risk score (0-100).
        model_probability: Optional ML model dropout probability.
        
    Returns:
        Dict with alert details including category, score, key factors.
    """
    risk_level = categorize_risk(risk_score)
    factors = _identify_risk_factors(student_row)
    
    # Emoji and urgency
    emoji_map = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
    
    student_name = student_row.get('Name', f"Student #{student_row.get('Student_ID', 'Unknown')}")
    
    alert = {
        'student_id': student_row.get('Student_ID', 'Unknown'),
        'student_name': student_name,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_emoji': emoji_map.get(risk_level, '⚪'),
        'model_probability': round(model_probability, 4) if model_probability is not None else None,
        'key_factors': factors,
        'recommended_actions': _get_recommendations(risk_level, factors),
        'semester': student_row.get('Semester', 'N/A')
    }
    
    return alert


def _get_recommendations(risk_level: str, factors: list) -> List[str]:
    """Generate intervention recommendations based on risk level and factors."""
    recommendations = []
    
    if risk_level == 'High':
        recommendations.append("🚨 URGENT: Schedule immediate counseling session")
        recommendations.append("📞 Contact student's guardian/parent")
    
    if risk_level in ('High', 'Medium'):
        recommendations.append("📋 Assign academic mentor for regular check-ins")
    
    # Factor-specific recommendations
    factor_names = [f['factor'] for f in factors]
    
    if any('attendance' in f.lower() for f in factor_names):
        recommendations.append("📍 Implement attendance monitoring with daily alerts")
    
    if any('gpa' in f.lower() for f in factor_names):
        recommendations.append("📚 Provide tutoring support and study group access")
    
    if any('engagement' in f.lower() or 'lms' in f.lower() for f in factor_names):
        recommendations.append("💻 Send personalized LMS engagement nudges")
    
    if any('stress' in f.lower() for f in factor_names):
        recommendations.append("🧘 Refer to campus wellness and mental health services")
    
    if any('financial' in f.lower() or 'tuition' in f.lower() for f in factor_names):
        recommendations.append("💰 Evaluate for emergency financial aid or scholarship")
    
    return recommendations[:5]


def batch_risk_assessment(
    df: pd.DataFrame,
    model_probabilities: Optional[np.ndarray] = None,
    student_id_col: str = 'Student_ID'
) -> dict:
    """Process an entire DataFrame and generate risk assessments.
    
    Uses the LATEST semester record for each student.
    
    Args:
        df: DataFrame with student records (may have multiple semesters).
        model_probabilities: Optional array of model probabilities aligned 
            with unique student IDs.
        student_id_col: Student identifier column.
        
    Returns:
        Dict with 'summary' (counts) and 'alerts' (list of alert dicts).
    """
    # Get latest semester per student
    latest = df.sort_values([student_id_col, 'Semester']).groupby(student_id_col).last().reset_index()
    
    alerts = []
    risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
    
    for i, (_, row) in enumerate(latest.iterrows()):
        student_dict = row.to_dict()
        
        model_prob = None
        if model_probabilities is not None and i < len(model_probabilities):
            model_prob = float(model_probabilities[i])
        
        risk_score = compute_risk_score(student_dict, model_prob)
        alert = generate_alert(student_dict, risk_score, model_prob)
        
        risk_level = alert['risk_level']
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        alerts.append(alert)
    
    # Sort by risk score (highest first)
    alerts.sort(key=lambda x: x['risk_score'], reverse=True)
    
    summary = {
        'total_students': len(latest),
        'high_risk': risk_counts.get('High', 0),
        'medium_risk': risk_counts.get('Medium', 0),
        'low_risk': risk_counts.get('Low', 0),
        'high_risk_pct': round(risk_counts.get('High', 0) / max(len(latest), 1) * 100, 1)
    }
    
    return {'summary': summary, 'alerts': alerts}


def print_risk_report(assessment: dict, top_n: int = 10) -> None:
    """Pretty-print a risk assessment report.
    
    Args:
        assessment: Output from batch_risk_assessment().
        top_n: Number of highest-risk students to display.
    """
    summary = assessment['summary']
    alerts = assessment['alerts']
    
    print("\n" + "=" * 70)
    print("  🏫 STUDENT DROPOUT EARLY WARNING REPORT")
    print("=" * 70)
    
    print(f"\n  Total Students Assessed: {summary['total_students']}")
    print(f"  ├── 🔴 High Risk:   {summary['high_risk']} ({summary['high_risk_pct']:.1f}%)")
    print(f"  ├── 🟡 Medium Risk: {summary['medium_risk']}")
    print(f"  └── 🟢 Low Risk:    {summary['low_risk']}")
    
    # Show top-N highest risk students
    high_risk_alerts = [a for a in alerts if a['risk_level'] == 'High'][:top_n]
    
    if high_risk_alerts:
        print(f"\n  {'─' * 66}")
        print(f"  TOP {min(top_n, len(high_risk_alerts))} HIGH-RISK STUDENTS")
        print(f"  {'─' * 66}")
        
        for alert in high_risk_alerts:
            prob_str = f" | Model: {alert['model_probability']:.1%}" if alert['model_probability'] else ""
            print(f"\n  {alert['risk_emoji']} Student: {alert['student_name']}")
            print(f"     Dropout Risk: {alert['risk_level'].upper()} ({alert['risk_score']:.0f}%){prob_str}")
            
            if alert['key_factors']:
                print(f"     Key Factors:")
                for factor in alert['key_factors'][:4]:
                    severity_icon = "🔴" if factor['severity'] == 'critical' else "🟡"
                    print(f"       {severity_icon} {factor['factor']}: {factor['value']}")
            
            if alert['recommended_actions']:
                print(f"     Actions:")
                for action in alert['recommended_actions'][:3]:
                    print(f"       → {action}")
    
    print("\n" + "=" * 70)

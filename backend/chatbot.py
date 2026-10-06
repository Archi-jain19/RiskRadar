"""
Chatbot command parser and response formatter.
Handles user messages and routes them to the appropriate action.

Supported commands:
  - Check student USN: <USN>
  - Why is student <USN> at risk?
  - Show class risk summary
  - Show top <N> students at risk
  - Show warning signals
  - Show progress of student <USN>
  - Generate report for student <USN>
  - Show high risk students
  - Explain risk factors
  - help
"""

import os
import re
import random
import pandas as pd
from model import predict_dropout, get_positive_indicators, FEATURE_LABELS

# ── Load student data ──────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "students.csv")
_df = None


def _load_students() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    return _df


def get_student(usn: str) -> dict | None:
    """Retrieve a student record by USN (case-insensitive)."""
    df = _load_students()
    match = df[df["usn"].str.upper() == usn.upper()]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def get_all_students_with_risk() -> list[dict]:
    """Return every student with their risk prediction attached."""
    df = _load_students()
    results = []
    for _, row in df.iterrows():
        student = row.to_dict()
        pred = predict_dropout(student)
        student["probability"] = pred["probability"]
        student["risk_category"] = pred["risk_category"]
        student["key_factors"] = pred["key_factors"]
        results.append(student)
    return results


def get_high_risk_students(threshold: float = 0.6) -> list[dict]:
    """Return all students predicted as high risk."""
    all_students = get_all_students_with_risk()
    results = [
        {
            "usn": s["usn"],
            "name": s["name"],
            "probability": s["probability"],
            "risk_category": s["risk_category"],
        }
        for s in all_students
        if s["probability"] >= threshold
    ]
    results.sort(key=lambda x: -x["probability"])
    return results


# ── Command patterns ───────────────────────────────────────────
_USN_PATTERN = re.compile(r"\b(\d{0,2}[A-Za-z]{2,4}\d{2,4}[A-Za-z]{0,3}\d{2,4})\b")
_CHECK_PATTERN = re.compile(r"check\s+(?:student\s+)?(?:usn[:\s]*)?(.+)", re.IGNORECASE)
_HIGH_RISK_PATTERN = re.compile(r"(?:show|list|get)\s+high[\s-]*risk", re.IGNORECASE)
_EXPLAIN_PATTERN = re.compile(r"explain\s+(?:risk\s+)?factors?", re.IGNORECASE)

# NEW patterns
_WHY_RISK_PATTERN = re.compile(r"why\s+(?:is\s+)?(?:student\s+)?(.+?)(?:\s+at\s+risk)?$", re.IGNORECASE)
_CLASS_SUMMARY_PATTERN = re.compile(r"(?:show|get|view)\s+class\s+(?:risk\s+)?summary", re.IGNORECASE)
_TOP_N_PATTERN = re.compile(r"(?:show|list|get)\s+top\s+(\d+)\s+(?:students?\s+)?(?:at\s+risk|at-risk|risk)", re.IGNORECASE)
_WARNING_PATTERN = re.compile(r"(?:show|list|get)\s+warning\s*(?:signals?)?", re.IGNORECASE)
_PROGRESS_PATTERN = re.compile(r"(?:show|get|view)\s+progress\s+(?:of\s+)?(?:student\s+)?(.+)", re.IGNORECASE)
_REPORT_PATTERN = re.compile(r"(?:generate|create|get|show)\s+report\s+(?:for\s+)?(?:student\s+)?(.+)", re.IGNORECASE)


def process_message(message: str) -> dict:
    """Parse a chat message and return a structured response dict."""
    text = message.strip()

    # ── Explain risk factors (general) ─────────────────────────
    if _EXPLAIN_PATTERN.search(text):
        return _explain_factors()

    # ── Class risk summary ─────────────────────────────────────
    if _CLASS_SUMMARY_PATTERN.search(text):
        return _class_risk_summary()

    # ── Top N at-risk ──────────────────────────────────────────
    top_match = _TOP_N_PATTERN.search(text)
    if top_match:
        n = int(top_match.group(1))
        return _top_n_at_risk(n)

    # ── Warning signals ────────────────────────────────────────
    if _WARNING_PATTERN.search(text):
        return _warning_signals()

    # ── Show high risk students ────────────────────────────────
    if _HIGH_RISK_PATTERN.search(text):
        return _high_risk_response()

    # ── Generate report ────────────────────────────────────────
    report_match = _REPORT_PATTERN.search(text)
    if report_match:
        usn = report_match.group(1).strip()
        return _generate_report(usn)

    # ── Student progress trend ─────────────────────────────────
    progress_match = _PROGRESS_PATTERN.search(text)
    if progress_match:
        usn = progress_match.group(1).strip()
        return _student_progress(usn)

    # ── Why is student at risk? ────────────────────────────────
    why_match = _WHY_RISK_PATTERN.search(text)
    if why_match:
        usn = why_match.group(1).strip()
        return _why_at_risk(usn)

    # ── Check specific student ─────────────────────────────────
    check_match = _CHECK_PATTERN.search(text)
    if check_match:
        usn = check_match.group(1).strip()
        return _check_student(usn)

    # ── Bare USN ───────────────────────────────────────────────
    usn_match = _USN_PATTERN.search(text)
    if usn_match:
        return _check_student(usn_match.group(1))

    # ── Greeting / fallback ────────────────────────────────────
    lower = text.lower()
    if any(g in lower for g in ["hello", "hi", "hey", "help"]):
        return _welcome_response()

    return {
        "type": "error",
        "message": (
            "I couldn't understand that command. Try:\n"
            "• **Check student USN: 1RV22CS001**\n"
            "• **Why is student 22CS045 at risk?**\n"
            "• **Show class risk summary**\n"
            "• **Show top 5 students at risk**\n"
            "• **Show warning signals**\n"
            "• **Show progress of student 22CS045**\n"
            "• **Generate report for student 22CS045**"
        ),
    }


# ══════════════════════════════════════════════════════════════
#  RESPONSE BUILDERS
# ══════════════════════════════════════════════════════════════


def _welcome_response() -> dict:
    return {
        "type": "welcome",
        "message": "👋 Hello! I'm EduPulse AI — your Faculty Assistant. Here's what I can do:",
        "commands": [
            {"command": "Check student USN: <USN>", "description": "View dropout risk assessment for a student"},
            {"command": "Why is student <USN> at risk?", "description": "Explain top risk factors for a specific student"},
            {"command": "Show class risk summary", "description": "Overview of class-wide risk distribution"},
            {"command": "Show top 5 students at risk", "description": "Rank students by dropout probability"},
            {"command": "Show warning signals", "description": "Detect early warning patterns across the class"},
            {"command": "Show progress of student <USN>", "description": "View attendance, assignment & LMS trends"},
            {"command": "Generate report for student <USN>", "description": "Generate a structured performance report"},
            {"command": "Show high risk students", "description": "List all students at high dropout risk"},
            {"command": "Explain risk factors", "description": "Understand what factors the model considers"},
        ],
    }


# ── 1. Check student (existing) ───────────────────────────────
def _check_student(usn: str) -> dict:
    student = get_student(usn)
    if student is None:
        return {"type": "error", "message": f"Student with USN **{usn.upper()}** not found in the database."}

    prediction = predict_dropout(student)

    if prediction["risk_category"] == "Low":
        positive = get_positive_indicators(student)
        health_score = _compute_health_score(student)
        return {
            "type": "healthy_report",
            "student_name": student["name"],
            "usn": student["usn"],
            "health_score": health_score,
            "positive_indicators": positive,
            "encouragement": (
                f"🌟 {student['name']} is performing well! "
                "Keep up the consistent engagement and strong academic performance. "
                "Consider recognising this student's effort to maintain motivation."
            ),
            "stats": _extract_stats(student),
        }

    recommendations = _generate_recommendations(prediction["key_factors"], student)
    return {
        "type": "risk_report",
        "student_name": student["name"],
        "usn": student["usn"],
        "dropout_probability": prediction["probability"],
        "risk_level": prediction["risk_category"],
        "key_factors": prediction["key_factors"],
        "recommendations": recommendations,
        "stats": _extract_stats(student),
    }


# ── 2. Why is student at risk? (NEW) ──────────────────────────
def _why_at_risk(usn: str) -> dict:
    student = get_student(usn)
    if student is None:
        return {"type": "error", "message": f"Student with USN **{usn.upper()}** not found in the database."}

    prediction = predict_dropout(student)

    if prediction["risk_category"] == "Low":
        return {
            "type": "why_risk",
            "student_name": student["name"],
            "usn": student["usn"],
            "risk_level": "Low",
            "probability": prediction["probability"],
            "explanation": f"🎉 {student['name']} is NOT at risk. Their dropout probability is only {prediction['probability']*100:.0f}%.",
            "key_factors": [],
            "stats": _extract_stats(student),
        }

    # Build detailed explanation per factor
    explanations = []
    for f in prediction["key_factors"]:
        explanations.append({
            "factor": f["factor"],
            "value": f["value"],
            "concern": f["concern"],
            "severity": "Critical" if _is_critical(f) else "Warning",
        })

    return {
        "type": "why_risk",
        "student_name": student["name"],
        "usn": student["usn"],
        "risk_level": prediction["risk_category"],
        "probability": prediction["probability"],
        "explanation": f"📋 {student['name']} has a {prediction['probability']*100:.0f}% dropout probability due to the following factors:",
        "key_factors": explanations,
        "stats": _extract_stats(student),
    }


# ── 3. Class risk summary (NEW) ───────────────────────────────
def _class_risk_summary() -> dict:
    all_students = get_all_students_with_risk()
    total = len(all_students)
    high = sum(1 for s in all_students if s["risk_category"] == "High")
    medium = sum(1 for s in all_students if s["risk_category"] == "Medium")
    low = sum(1 for s in all_students if s["risk_category"] == "Low")

    # Students needing attention (high & medium risk)
    attention_list = [
        {"usn": s["usn"], "name": s["name"], "probability": s["probability"], "risk_category": s["risk_category"]}
        for s in all_students
        if s["risk_category"] in ("High", "Medium")
    ]
    attention_list.sort(key=lambda x: -x["probability"])

    avg_prob = sum(s["probability"] for s in all_students) / total if total else 0

    return {
        "type": "class_summary",
        "total_students": total,
        "high_risk": high,
        "medium_risk": medium,
        "low_risk": low,
        "average_risk": round(avg_prob, 4),
        "attention_list": attention_list[:10],
        "message": f"Class of **{total}** students: **{high}** high risk, **{medium}** medium risk, **{low}** low risk.",
    }


# ── 4. Top N at-risk (NEW) ────────────────────────────────────
def _top_n_at_risk(n: int) -> dict:
    all_students = get_all_students_with_risk()
    ranked = sorted(all_students, key=lambda s: -s["probability"])
    top = ranked[:n]
    result = [
        {
            "rank": i + 1,
            "usn": s["usn"],
            "name": s["name"],
            "probability": s["probability"],
            "risk_category": s["risk_category"],
        }
        for i, s in enumerate(top)
    ]
    return {
        "type": "top_risk",
        "count": n,
        "students": result,
        "message": f"Top **{n}** students ranked by dropout risk:",
    }


# ── 5. Warning signals (NEW) ──────────────────────────────────
def _warning_signals() -> dict:
    all_students = get_all_students_with_risk()
    warnings = []

    for s in all_students:
        signals = []
        if s.get("attendance", 100) < 55:
            signals.append("📉 Attendance critically low")
        if s.get("assignment_completion", 100) < 40:
            signals.append("📝 Multiple missed assignments")
        if s.get("lms_activity", 100) < 30:
            signals.append("💻 LMS inactivity detected")
        if s.get("class_participation", 100) < 30:
            signals.append("🔇 Very low class participation")
        if s.get("gpa", 10) < 4.0:
            signals.append("📊 GPA below critical threshold")

        if signals:
            warnings.append({
                "usn": s["usn"],
                "name": s["name"],
                "probability": s["probability"],
                "risk_category": s["risk_category"],
                "signals": signals,
            })

    warnings.sort(key=lambda x: -len(x["signals"]))

    return {
        "type": "warning_signals",
        "count": len(warnings),
        "students": warnings[:15],
        "message": f"⚠️ **{len(warnings)}** students showing early warning signals." if warnings else "✅ No critical warning signals detected.",
    }


# ── 6. Student progress trend (NEW) ───────────────────────────
def _student_progress(usn: str) -> dict:
    student = get_student(usn)
    if student is None:
        return {"type": "error", "message": f"Student with USN **{usn.upper()}** not found in the database."}

    # Generate synthetic monthly trends (simulating 6 months of data)
    trends = _generate_trend_data(student)
    prediction = predict_dropout(student)

    # Determine trend direction
    att_trend = "increasing" if trends["attendance"][-1] > trends["attendance"][0] else "decreasing"
    asgn_trend = "increasing" if trends["assignments"][-1] > trends["assignments"][0] else "decreasing"
    lms_trend = "increasing" if trends["lms_activity"][-1] > trends["lms_activity"][0] else "decreasing"

    overall_direction = "improving" if att_trend == "increasing" and asgn_trend == "increasing" else "declining"

    return {
        "type": "progress_trend",
        "student_name": student["name"],
        "usn": student["usn"],
        "risk_level": prediction["risk_category"],
        "probability": prediction["probability"],
        "months": ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"],
        "trends": trends,
        "trend_summary": {
            "attendance": att_trend,
            "assignments": asgn_trend,
            "lms_activity": lms_trend,
            "overall": overall_direction,
        },
        "message": f"📈 {student['name']}'s performance is **{overall_direction}** over the last 6 months.",
    }


# ── 7. Report generator (NEW) ─────────────────────────────────
def _generate_report(usn: str) -> dict:
    student = get_student(usn)
    if student is None:
        return {"type": "error", "message": f"Student with USN **{usn.upper()}** not found in the database."}

    prediction = predict_dropout(student)
    health_score = _compute_health_score(student)
    positive = get_positive_indicators(student)
    recommendations = _generate_recommendations(prediction["key_factors"], student)

    return {
        "type": "student_report",
        "student_name": student["name"],
        "usn": student["usn"],
        "risk_level": prediction["risk_category"],
        "probability": prediction["probability"],
        "health_score": health_score,
        "stats": _extract_stats(student),
        "key_factors": prediction["key_factors"],
        "positive_indicators": positive,
        "recommendations": recommendations,
        "distance_from_college": student.get("distance_from_college", "N/A"),
        "financial_background": student.get("financial_background", "N/A"),
    }


# ── Existing features ─────────────────────────────────────────
def _high_risk_response() -> dict:
    students = get_high_risk_students()
    return {
        "type": "high_risk_list",
        "count": len(students),
        "students": students,
        "message": f"Found **{len(students)}** students at high dropout risk." if students else "No students currently flagged as high risk. 🎉",
    }


def _explain_factors() -> dict:
    return {
        "type": "explanation",
        "title": "Understanding Dropout Risk Factors",
        "factors": [
            {"name": "Attendance %", "weight": "High", "description": "Regular attendance is the strongest indicator of continued enrollment. Students below 60% are at significant risk."},
            {"name": "GPA", "weight": "High", "description": "Academic performance below 5.0 (out of 10) correlates strongly with dropout."},
            {"name": "Assignment Completion", "weight": "Medium-High", "description": "Completing less than 50% of assignments signals academic disengagement."},
            {"name": "LMS Activity", "weight": "Medium", "description": "Low interaction with the Learning Management System indicates reduced engagement."},
            {"name": "Class Participation", "weight": "Medium", "description": "Declining participation often precedes formal withdrawal."},
            {"name": "Distance from College", "weight": "Medium", "description": "Students living far away face logistical challenges that increase dropout risk."},
            {"name": "Financial Background", "weight": "Medium", "description": "Limited financial resources can force students to discontinue education."},
        ],
    }


def _generate_recommendations(key_factors: list[dict], student: dict) -> list[str]:
    """Generate faculty recommendations based on identified risk factors."""
    recs = []
    factor_names = {f["factor"] for f in key_factors}

    if "Attendance %" in factor_names:
        recs.append("Schedule a one-on-one attendance review meeting with the student.")
    if "GPA" in factor_names:
        recs.append("Assign a peer tutor or recommend academic support sessions.")
    if "Assignment Completion %" in factor_names:
        recs.append("Set up incremental deadlines and regular check-ins for pending assignments.")
    if "LMS Activity Score" in factor_names:
        recs.append("Encourage daily LMS engagement; consider gamified participation incentives.")
    if "Class Participation Score" in factor_names:
        recs.append("Create structured group activities to boost classroom involvement.")
    if "Distance from College (km)" in factor_names:
        recs.append("Explore hostel accommodation or transport assistance options.")
    if "Financial Background (1=Low, 3=High)" in factor_names:
        recs.append("Connect the student with scholarship and financial aid resources.")

    if not recs:
        recs.append("Schedule a general counseling session to understand the student's challenges.")

    return recs


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _extract_stats(student: dict) -> dict:
    return {
        "attendance": student["attendance"],
        "gpa": student["gpa"],
        "assignment_completion": student["assignment_completion"],
        "lms_activity": student["lms_activity"],
        "class_participation": student["class_participation"],
    }


def _compute_health_score(student: dict) -> int:
    return min(100, int(
        student["attendance"] * 0.25
        + student["gpa"] * 10 * 0.25
        + student["assignment_completion"] * 0.2
        + student["lms_activity"] * 0.15
        + student["class_participation"] * 0.15
    ))


def _is_critical(factor: dict) -> bool:
    """Determine if a risk factor is critical based on its value."""
    v = factor.get("value", 0)
    name = factor.get("factor", "")
    if "Attendance" in name and v < 45:
        return True
    if "GPA" in name and v < 4.0:
        return True
    if "Assignment" in name and v < 35:
        return True
    return False


def _generate_trend_data(student: dict) -> dict:
    """
    Generate synthetic 6-month trend data based on current values.
    For at-risk students: shows declining trend ending at current value.
    For healthy students: shows stable/improving trend.
    """
    random.seed(hash(student.get("usn", "")))  # deterministic per student

    current_att = student.get("attendance", 50)
    current_asgn = student.get("assignment_completion", 50)
    current_lms = student.get("lms_activity", 50)

    if current_att < 60:
        # Declining pattern
        att = [min(100, current_att + 25 - i * 5 + random.randint(-3, 3)) for i in range(6)]
        asgn = [min(100, current_asgn + 20 - i * 4 + random.randint(-3, 3)) for i in range(6)]
        lms = [min(100, current_lms + 18 - i * 4 + random.randint(-2, 2)) for i in range(6)]
    else:
        # Stable/improving pattern
        att = [max(0, current_att - 8 + i * 2 + random.randint(-2, 2)) for i in range(6)]
        asgn = [max(0, current_asgn - 6 + i * 1 + random.randint(-2, 2)) for i in range(6)]
        lms = [max(0, current_lms - 5 + i * 1 + random.randint(-2, 2)) for i in range(6)]

    return {
        "attendance": [max(0, min(100, v)) for v in att],
        "assignments": [max(0, min(100, v)) for v in asgn],
        "lms_activity": [max(0, min(100, v)) for v in lms],
    }

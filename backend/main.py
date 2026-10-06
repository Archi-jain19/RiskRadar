"""
DropGuard AI — Unified FastAPI Backend
=======================================
Combines:
  - CSV upload & batch prediction (from dashboard branch)
  - Chatbot NLP endpoint (from ai-faculty-copilot branch)
  - Individual student prediction & lookup
Connected to the trained v3 ensemble model.
"""

import sys
import os
import io

if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from model import predict_dropout_from_raw, predict_dropout_batch, get_positive_indicators, _identify_key_factors
from chatbot import process_message, get_student, get_high_risk_students, get_all_students_with_risk

# In-memory store for recent batch uploads when Supabase is not configured
_latest_uploaded_students: list[dict] = []

app = FastAPI(title="DropGuard AI — Faculty Analytics Engine", version="2.0.0")

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase (optional) ──────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Successfully connected to Supabase!")
    except Exception as e:
        print(f"❌ Supabase Connection Failed: {e}")
else:
    print("⚠️  Supabase not configured (optional). Running with local model only.")


# ── Request / Response models ─────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: dict

class PredictRequest(BaseModel):
    """Single student prediction from raw features."""
    Age: float = 20
    Gender: str = "Male"
    Family_Income: float = 50000
    Internet_Access: str = "Yes"
    Study_Hours_per_Day: float = 3.0
    Attendance_Rate: float = 75.0
    Assignment_Delay_Days: float = 2
    Travel_Time_Minutes: float = 30
    Part_Time_Job: str = "No"
    Scholarship: str = "No"
    Stress_Index: float = 5.0
    GPA: float = 2.5
    Semester_GPA: float = 2.5
    CGPA: float = 2.5
    Semester: str = "Year 2"
    Department: str = "CS"
    Parental_Education: str = "Bachelor"


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "online", "message": "DropGuard AI Analytics Engine is running.", "version": "2.0.0"}


@app.post("/api/upload-student-data")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file with student records. Runs the v3 ensemble model
    on each row and returns predictions with risk scores.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Run real model predictions
        results = predict_dropout_batch(df)

        # Save to normalized Supabase tables
        if supabase:
            try:
                # 1. First get/upsert students
                for _, row in results.iterrows():
                    usn = str(row.get("Student_ID", row.get("usn", row.get("student_id", "")))).strip()
                    if not usn:
                        continue

                    # 1. Upsert into students table
                    student_record = {
                        "usn": usn,
                        "name": str(row.get("name", "")) or None,
                        "age": int(row["Age"]) if "Age" in row and pd.notna(row.get("Age")) else None,
                        "gender": str(row.get("Gender", "")) or None,
                        "department": str(row.get("Department", row.get("program", ""))) or None,
                        "semester": str(row.get("Semester", "")) or None,
                        "parental_education": str(row.get("Parental_Education", "")) or None,
                        "family_income": float(row["Family_Income"]) if "Family_Income" in row and pd.notna(row.get("Family_Income")) else None,
                        "internet_access": str(row.get("Internet_Access", "")).lower() == "yes" if "Internet_Access" in row else None,
                        "part_time_job": str(row.get("Part_Time_Job", "")).lower() == "yes" if "Part_Time_Job" in row else None,
                        "scholarship": str(row.get("Scholarship", "")).lower() == "yes" if "Scholarship" in row else None,
                    }
                    student_record = {k: v for k, v in student_record.items() if pd.notna(v) and v is not None}
                    student_record["usn"] = usn
                    
                    resp = supabase.table("students").upsert(student_record, on_conflict="usn").execute()
                    student_id = resp.data[0]["id"]

                    # 2. Insert metrics snapshot
                    metrics_record = {
                        "student_id": student_id,
                        "attendance": float(row.get("Attendance_Rate", row.get("attendance", 0)) or 0),
                        "gpa": float(row.get("GPA", row.get("gpa", 0)) or 0),
                        "semester_gpa": float(row.get("Semester_GPA", 0) or 0) if "Semester_GPA" in row else None,
                        "cgpa": float(row.get("CGPA", 0) or 0) if "CGPA" in row else None,
                        "assignment_completion": float(row.get("assignment_completion", 0) or 0) if "assignment_completion" in row else None,
                        "assignment_delay_days": float(row.get("Assignment_Delay_Days", 0) or 0) if "Assignment_Delay_Days" in row else None,
                        "lms_activity": float(row.get("lms_activity", 0) or 0) if "lms_activity" in row else None,
                        "class_participation": float(row.get("class_participation", 0) or 0) if "class_participation" in row else None,
                        "study_hours_per_day": float(row.get("Study_Hours_per_Day", 0) or 0) if "Study_Hours_per_Day" in row else None,
                        "stress_index": float(row.get("Stress_Index", 0) or 0) if "Stress_Index" in row else None,
                        "distance_from_college": float(row.get("distance_from_college", 0) or 0) if "distance_from_college" in row else None,
                        "travel_time_minutes": float(row.get("Travel_Time_Minutes", 0) or 0) if "Travel_Time_Minutes" in row else None,
                        "financial_background": int(row.get("financial_background", 0) or 0) if "financial_background" in row else None,
                    }
                    metrics_record = {k: v for k, v in metrics_record.items() if pd.notna(v) and v is not None}
                    metrics_record["student_id"] = student_id
                    supabase.table("student_metrics").insert(metrics_record).execute()

                    # 3. Insert prediction with risk factors
                    factors = _identify_key_factors(row.fillna(0).to_dict(), float(row.get("risk_score", 0)))
                    prediction_record = {
                        "student_id": student_id,
                        "risk_score": float(row.get("risk_score", 0)),
                        "risk_level": str(row.get("risk_level", "Unknown")),
                        "model_version": "v3",
                        "risk_factors": factors
                    }
                    supabase.table("predictions").insert(prediction_record).execute()
                    
                print(f"Saved {len(results)} records to normalized Supabase tables.")
            except Exception as e:
                print(f"Supabase insert failed: {e}")
                import traceback
                traceback.print_exc()

        # Return as JSON
        records = results.fillna(0).to_dict(orient='records')
        global _latest_uploaded_students
        _latest_uploaded_students = records
        return {
            "status": "success",
            "message": f"Successfully evaluated {len(records)} student profiles using v3 ensemble model.",
            "data": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/student-predictions")
async def get_student_predictions():
    """Fetch all students with their latest metrics and predictions from Normalized schema (or local dataset fallback)"""
    try:
        if not supabase:
            if _latest_uploaded_students:
                return {"status": "success", "data": _latest_uploaded_students}

            # Fallback to local students.csv dataset with predictions
            students = get_all_students_with_risk()
            flat_data = []
            for s in students:
                item = dict(s)
                item["risk_score"] = item.get("probability", 0)
                item["risk_level"] = item.get("risk_category", "Unknown")
                item["Student_ID"] = item.get("usn")
                item["Department"] = item.get("program", "Computer Science")
                item["Attendance_Rate"] = item.get("attendance", 0)
                item["GPA"] = item.get("gpa", 0)
                flat_data.append(item)
            return {"status": "success", "data": flat_data}
            
        resp = supabase.table("students").select("*, predictions(*), student_metrics(*)").execute()
        
        flat_data = []
        for s in resp.data:
            pred = s.get("predictions", [])
            met = s.get("student_metrics", [])
            latest_pred = pred[0] if pred else {}
            latest_met = met[0] if met else {}
            
            combined = {**s, **latest_met, **latest_pred}
            combined.pop("predictions", None)
            combined.pop("student_metrics", None)
            combined["student_id"] = s.get("id") 
            flat_data.append(combined)
            
        return {"status": "success", "data": flat_data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=dict)
async def predict_single(req: PredictRequest):
    """Predict dropout risk for a single student from raw features."""
    student_data = req.model_dump()
    try:
        result = predict_dropout_from_raw(student_data)
        return {"status": "success", "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chatbot Endpoints ─────────────────────────────────────────

def format_chat_response(res: dict) -> str:
    rtype = res.get("type")
    
    if rtype == "error":
        return f"⚠️ {res.get('message', 'Error')}"
        
    if rtype == "welcome":
        msgs = [res.get("message", "")]
        for cmd in res.get("commands", []):
            msgs.append(f"• <b>{cmd['command']}</b>: {cmd['description']}")
        return "<br>".join(msgs)
        
    if rtype == "healthy_report":
        msgs = [f"✅ <b>{res.get('student_name')}</b> ({res.get('usn')}) is performing well!"]
        msgs.append(f"<b>Health Score</b>: {res.get('health_score')}/100")
        if res.get("positive_indicators"):
            msgs.append("<br><b>Positive Indicators:</b>")
            for ind in res.get("positive_indicators"):
                msgs.append(f"• {ind['indicator']}: {ind['value']}")
        msgs.append(f"<br><i>{res.get('encouragement')}</i>")
        return "<br>".join(msgs)
        
    if rtype == "risk_report":
        msgs = [f"⚠️ <b>{res.get('student_name')}</b> ({res.get('usn')}) is at <b>{res.get('risk_level')}</b> risk."]
        msgs.append(f"<b>Dropout Probability</b>: {res.get('dropout_probability', 0)*100:.0f}%")
        if res.get("key_factors"):
            msgs.append("<br><b>Key Risk Factors:</b>")
            for f in res.get("key_factors"):
                msgs.append(f"• {f.get('factor')}: {f.get('value')} ({f.get('concern')})")
        if res.get("recommendations"):
            msgs.append("<br><b>Recommendations:</b>")
            for r in res.get("recommendations"):
                msgs.append(f"• {r}")
        return "<br>".join(msgs)
        
    if rtype == "why_risk":
        msgs = [res.get("explanation", "")]
        if res.get("key_factors"):
            for f in res.get("key_factors"):
                msgs.append(f"• <b>{f.get('factor')}</b>: {f.get('value')} ({f.get('severity')})")
        return "<br>".join(msgs)
        
    if rtype == "class_summary":
        msgs = [res.get("message", "")]
        if res.get("attention_list"):
            msgs.append("<br><b>Top Students Needing Attention:</b>")
            for s in res.get("attention_list")[:5]:
                msgs.append(f"• {s['name']} ({s['usn']}): {s['risk_category']} Risk")
        return "<br>".join(msgs)
        
    if rtype == "top_risk":
        msgs = [res.get("message", "")]
        for s in res.get("students"):
            msgs.append(f"{s['rank']}. {s['name']} ({s['usn']}) - {s['risk_category']} Risk ({s['probability']*100:.0f}%)")
        return "<br>".join(msgs)
        
    if rtype == "warning_signals":
        msgs = [res.get("message", "")]
        for s in res.get("students")[:5]:
            msgs.append(f"<br><b>{s['name']}</b> ({s['usn']}):")
            for sig in s.get("signals", []):
                msgs.append(f"• {sig}")
        return "<br>".join(msgs)
        
    if rtype == "progress_trend":
        msgs = [res.get("message", "")]
        ts = res.get("trend_summary", {})
        msgs.append(f"• Attendance: {ts.get('attendance')}")
        msgs.append(f"• Assignments: {ts.get('assignments')}")
        msgs.append(f"• LMS Activity: {ts.get('lms_activity')}")
        return "<br>".join(msgs)
        
    if rtype == "student_report":
        msgs = [f"📄 <b>Report for {res.get('student_name')}</b> ({res.get('usn')})"]
        msgs.append(f"<b>Risk Level</b>: {res.get('risk_level')} ({res.get('probability', 0)*100:.0f}%)")
        msgs.append(f"<b>Health Score</b>: {res.get('health_score')}/100")
        return "<br>".join(msgs)

    if rtype == "high_risk_list":
        msgs = [str(res.get("message", ""))]
        for s in res.get("students"):
            msgs.append(f"• {s['name']} ({s['usn']}): {s['probability']*100:.0f}% risk")
        return "<br>".join(msgs)
        
    if rtype == "explanation":
        msgs = [f"<b>{res.get('title', 'Factors')}</b>"]
        for f in res.get("factors", []):
            msgs.append(f"• <b>{f['name']}</b> ({f['weight']}): {f['description']}")
        return "<br>".join(msgs)

    return str(res.get("message", "Processed successfully."))

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """React frontend chatbot endpoint that returns a simple string reply."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    result = process_message(req.message)
    formatted_reply = format_chat_response(result)
    return {"reply": formatted_reply, "raw": result}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chatbot endpoint — accepts a user message and returns a structured response."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    result = process_message(req.message)
    return ChatResponse(response=result)


@app.get("/students/high-risk")
async def high_risk():
    """Return all students flagged as high dropout risk."""
    return get_high_risk_students()


@app.get("/students/{usn}")
async def student_detail(usn: str):
    """Return raw student data for a given USN."""
    student = get_student(usn)
    if student is None:
        raise HTTPException(status_code=404, detail=f"Student {usn} not found.")
    return student

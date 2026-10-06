import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    print("NO ENV VARS")
    exit()

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    res = supabase.table('student_predictions').select('*').limit(5).execute()
    print("Fetched successfully. Num records:", len(res.data))
    if len(res.data) > 0:
        print(res.data[0])
except Exception as e:
    print("Fetch Error:", e)

# Test insert
try:
    test_record = {"student_id": "TEST01", "risk_score": 0.5, "risk_level": "Medium", "gpa": 3.0, "attendance": 80, "program": "Test"}
    res2 = supabase.table('student_predictions').insert(test_record).execute()
    print("Insert successfully. Data:", res2.data)
except Exception as e:
    print("Insert Error:", e)

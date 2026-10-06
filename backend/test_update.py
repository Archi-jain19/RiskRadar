import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    print("Testing Update...")
    db_id = "2de02e49-e727-44fc-8aca-e12fe732df3e"
    update_data = {"gpa": 3.5, "attendance": 90.0, "program": "Engineering"}
    response = supabase.table('student_predictions').update(update_data).eq('id', db_id).execute()
    print("Response:", response)
except Exception as e:
    import traceback
    print("Exception!!")
    traceback.print_exc()

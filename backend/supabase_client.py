"""
Supabase Client Module
======================
Centralised Supabase connection that reads configuration from the .env file.

Usage:
    from supabase_client import supabase, is_connected

Environment variables required (in .env at project root):
    SUPABASE_URL  – Your Supabase project URL
    SUPABASE_KEY  – Your Supabase anon/public key
"""

import sys
import os
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load from the project-root .env (one level up from /backend)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY")

supabase = None
is_connected: bool = False

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        is_connected = True
        print("✅ Supabase connected successfully!")
    except ImportError:
        print("❌ supabase-py is not installed. Run: pip install supabase")
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
else:
    print("⚠️  SUPABASE_URL or SUPABASE_KEY not found in .env — Supabase disabled.")

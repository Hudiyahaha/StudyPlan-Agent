"""Tests run offline on the mock policy with a pinned clock (Monday morning), whatever .env contains."""
import os

os.environ["MODEL_PROVIDER"] = "mock"
os.environ["MODEL_NAME"] = "studyplan-mock-v1"
os.environ["FALLBACK_MODELS"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["REFERENCE_DATE"] = "2026-10-05T08:00:00"

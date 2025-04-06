# backend/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Nessie & Gemini keys
NESSIE_API_KEY = os.getenv("NESSIE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# If you need Clerk secret key on the backend for verifying tokens:
# CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
import os
from dotenv import load_dotenv

load_dotenv()

# Gemini key (used for customer profile generation & LLM calls)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# If you need Clerk secret key on the backend for verifying tokens:
# CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
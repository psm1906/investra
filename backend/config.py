import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration for API keys (accessible throughout the app)
NESSIE_API_KEY = os.getenv("NESSIE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

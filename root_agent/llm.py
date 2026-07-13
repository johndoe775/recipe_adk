import os
from google.genai import Client
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = Client(api_key=api_key)

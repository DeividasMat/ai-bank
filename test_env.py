import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"API key loaded: {'Yes' if api_key else 'No'}")
print(f"API key starts with: {api_key[:7]}..." if api_key else "No key found")


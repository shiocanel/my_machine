import os
from dotenv import load_dotenv

load_dotenv()

keys = os.getenv("OPENAI_API_KEY")

print("API keys loaded")
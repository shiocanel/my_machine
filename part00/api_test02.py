import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

print("Sending rquest to cloud... Please wait!")

response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "Explain matrix multiplication in one simple sentence."}
    ]
)

print("--Repsone Received--")
print(response.choices[0].message.content)
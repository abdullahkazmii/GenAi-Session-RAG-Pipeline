from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv
import pydantic

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = """
Extract the name, age, and city from this sentence:
"John is 25 years old and lives in Berlin."
Respond in JSON format.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(response_mime_type="application/json")
)
print("Response======>", response)
print(response.text)

try:
    data = json.loads(response.text)
    print("Parsed JSON:", data)
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Raw output:", response.text)


class Person(pydantic.BaseModel):
    name: str
    age: int
    city: str

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Person
    )
)

try:
    data = json.loads(response.text)
    print("Parsed JSON with Schema:", data)
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Raw output:", response.text)

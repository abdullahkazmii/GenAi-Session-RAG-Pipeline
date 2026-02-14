# Real-time token streaming
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response_stream = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Tell me about riphah.",
)

for chunk in response_stream:
    print(chunk.text, end="", flush=True)
print()

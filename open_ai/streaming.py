# Real-time token streaming
from openai import OpenAI
import os
from dotenv import load_dotenv


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)

stream = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Tell me about dev weekends"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
print()

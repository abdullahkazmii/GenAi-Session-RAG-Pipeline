from openai import OpenAI
import os
from dotenv import load_dotenv


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ],
    )

    print("Assistant:", response.choices[0].message.content)
    print("---")

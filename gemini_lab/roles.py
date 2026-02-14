from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant."
    ),
    history=[]
)
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = chat.send_message(user_input)

    print("Assistant:", response.text)
    print("---")

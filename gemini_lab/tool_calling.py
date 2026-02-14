from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_current_weather(location: str, unit: str = "celsius"):
    return {"location": location, "temperature": "20", "unit": unit, "condition": "Sunny"}

def calculate_expression(expression: str, precision: int = 2):
    try:
        return str(round(eval(expression), precision))
    except:
        return "Error"

tools = [get_current_weather, calculate_expression]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is (25 + 7) divided by 4?",
    config=types.GenerateContentConfig(
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
    )
)
print("Message============> ", response.text)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the weather in London?",
    config=types.GenerateContentConfig(
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
    )
)
print("Message============> ", response.text)

from openai import OpenAI
import json

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)


# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "precision": {"type": "integer", "default": 2},
                },
                "required": ["expression"],
            },
        },
    },
]

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "What is (25 + 7) divided by 4?"}],
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message
print("Message============> ", msg)

# Handle tool calls
if msg.tool_calls:
    for tool in msg.tool_calls:
        args = json.loads(tool.function.arguments)
        print(f"Tool Need to run: {tool.function.name}")
        print(f"Args: {args}")
else:
    print("💬 Model said:", msg.content)

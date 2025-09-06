from openai import OpenAI
import json
import os
from dotenv import load_dotenv


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)

prompt = """
Extract the name, age, and city from this sentence:
"John is 25 years old and lives in Berlin."
Respond in JSON format.
"""

response = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
)

# Parse and print result
try:
    data = json.loads(response.choices[0].message.content)
    print("Parsed JSON:", data)
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Raw output:", response.choices[0].message.content)


### Structured Response

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                },
                "required": ["name", "age", "city"],
                "additionalProperties": False,
            },
        },
    },
)

# Parse and print result
try:
    data = json.loads(response.choices[0].message.content)
    print("Parsed JSON:", data)
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Raw output:", response.choices[0].message.content)

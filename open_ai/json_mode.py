from openai import OpenAI
import json

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
    
)

prompt = """
Extract the name, age, and city from this sentence:
"John is 25 years old and lives in Berlin."
Respond in JSON format.
"""

response = client.chat.completions.create(
    model="openai/gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ],
    response_format={"type": "json_object"}
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
    messages=[
        {"role": "user", "content": prompt}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"}
                },
                "required": ["name", "age", "city"],
                "additionalProperties": False
            }
        }
    }
)

# Parse and print result
try:
    data = json.loads(response.choices[0].message.content)
    print("Parsed JSON:", data)
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Raw output:", response.choices[0].message.content)

from openai import OpenAI


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hi, What's is Dev Weekends ?"}],
)

# print response
print("Response======>", response)
print(response.choices[0].message.content)

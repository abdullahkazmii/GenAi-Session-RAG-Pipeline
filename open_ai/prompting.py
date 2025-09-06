from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

print("🧠 PROMPTING TECHNIQUES DEMO\n")
print("=" * 60)


examples = {

    # ========================================
    # 1. ZERO-SHOT: No examples, just ask
    # ========================================
    "zero_shot": {
        "prompt": "A bakery sells cupcakes for $2 each. If Sarah buys 7, how much does she spend?",
        "explanation": "No guidance — model must infer how to answer."
    },

    # ========================================
    # 2. FEW-SHOT: Show examples before asking
    # ========================================
    "few_shot": {
        "prompt": """Cost Calculation Examples:
                    Item: Apple | Price: $1 | Quantity: 5 | Total: $5
                    Item: Book  | Price: $12| Quantity: 3 | Total: $36

                    Now calculate:
                    Item: Mug   | Price: $4 | Quantity: 6 | Total: ?""",
        "explanation": "Model learns pattern from examples before applying it."
    },

    # ========================================
    # 3. CHAIN-OF-THOUGHT (CoT): Think step-by-step
    # ========================================
    "chain_of_thought": {
        "prompt": """A store has 100 bottles. They sell 24 in the morning and 37 in the afternoon.
                    Then they receive a shipment of 50 more.
                    How many bottles are left at the end of the day?

                    Think step by step.""",
        "explanation": "Encourages internal reasoning — 'show your work' for LLMs."
    },

    # ========================================
    # 4. ReAct (Reason + Act): Simulate tool use
    # ========================================
    "react": {
        "prompt": """Question: When was the Hubble Space Telescope launched?

                    Thought: I don't know the exact date. I should look it up.
                    Action: search("Hubble Space Telescope launch date")
                    Observation: The Hubble Space Telescope was launched on April 24, 1990.
                    Thought: I found the answer.
                    Answer: April 24, 1990.""",
        "explanation": "Combines reasoning (Thought) with actions (Action) — foundation for AI agents."
    }
}

# Run each example with clear labeling
for method, data in examples.items():
    print(f"\n🔹 TECHNIQUE: {method.upper()}")
    print(f"📝 Explanation: {data['explanation']}")
    print(f"💬 Prompt: {data['prompt']}")
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": data["prompt"]}
            ],
            max_tokens=150
        )
        answer = response.choices[0].message.content
        print(f"✅ Model Response: {answer}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("-" * 60)
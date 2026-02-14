from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("PROMPTING TECHNIQUES\n")
print("=" * 60)


examples = {
    # ========================================
    # 1. ZERO-SHOT: No examples, just ask
    # ========================================
    "zero_shot": {
        "prompt": "A bakery sells cupcakes for $2 each. If Sarah buys 7, how much does she spend?",
        "explanation": "No guidance — model must infer how to answer.",
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
        "explanation": "Model learns pattern from examples before applying it.",
    },
    # ========================================
    # 3. CHAIN-OF-THOUGHT (CoT): Think step-by-step
    # ========================================
    "chain_of_thought": {
        "prompt": """A store has 100 bottles. They sell 24 in the morning and 37 in the afternoon.
                    Then they receive a shipment of 50 more.
                    How many bottles are left at the end of the day?

                    Think step by step.""",
        "explanation": "Encourages internal reasoning — 'show your work' for LLMs.",
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
        "explanation": "Combines reasoning (Thought) with actions (Action) — foundation for AI agents.",
    },
}

for method, data in examples.items():
    print(f"Technique: {method.upper()}")
    print(f"Explanation: {data['explanation']}")
    print(f"Prompt: {data['prompt']}")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=data["prompt"],
            config=types.GenerateContentConfig(max_output_tokens=2000)
        )
        answer = response.text
        print(f"Model Response: {answer}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 60)

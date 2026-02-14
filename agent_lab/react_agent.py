import os
import getpass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API key is set
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
# Langchain Google GenAI expects this env var or passed explicitly
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# 1. Define Tools
@tool
def calculator(expression: str) -> str:
    """Calculates the result of a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(city: str) -> str:
    """Gets the current weather for a given city."""
    # Mock data for demo purposes
    weather_data = {
        "london": "Rainy, 15°C",
        "new york": "Sunny, 22°C",
        "tokyo": "Cloudy, 18°C",
        "paris": "Sunny, 20°C"
    }
    return weather_data.get(city.lower(), "Weather data not available for this city.")

tools = [calculator, get_weather]

# 2. Initialize Model
# We use gemini-2.5-flash for speed and tool calling capabilities
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Create Agent
# create_react_agent is a prebuilt helper that sets up the graph:
# Model -> Tools -> Model loop
agent_executor = create_react_agent(llm, tools)

# 4. Run Agent
def run_demo(query: str):
    print(f"\nUser: {query}")
    print("-" * 50)
    
    # helper to print streaming chunks
    for chunk in agent_executor.stream({"messages": [("user", query)]}, stream_mode="values"):
        # The last value in the stream is the final state
        # We can inspect messages to see the thought process
        messages = chunk["messages"]
        last_message = messages[-1]
        
        # Pretty print based on message type
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                print(f"🤖 Agent decides to call tool: {tool_call['name']} with args: {tool_call['args']}")
        elif last_message.type == "tool":
             print(f"🔧 Tool Output: {last_message.content}")
        elif last_message.type == "ai":
             print(f"💡 Agent: {last_message.content}")

if __name__ == "__main__":
    print("🤖 LangGraph ReAct Agent Demo")
    
    # Test 1: Math
    run_demo("Calculate 15 * 4 + 10")
    
    # Test 2: Weather (Tool use)
    run_demo("What is the weather in London?")
    
    # Test 3: Complex (Reasoning)
    run_demo("What is the weather in New York? And if I multiply the temperature by 2, what do I get?")

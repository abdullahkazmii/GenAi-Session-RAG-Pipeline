import os
import getpass
from dotenv import load_dotenv
from typing import List, Annotated
import operator
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ==========================================
# 1. Define Tools
# ==========================================
@tool
def search_tool(query: str):
    """Searches for information."""
    return f"Result for {query}: [Mock Info related to {query}]"

@tool
def calculate_tool(expression: str):
    """Calculates a math expression."""
    return f"Calculation result: {eval(expression)}"

tools = {
    "search": search_tool,
    "calculate": calculate_tool
}

# ==========================================
# 2. Define State
# ==========================================
class PlannerState(TypedDict):
    objective: str
    plan: List[str]
    past_steps: Annotated[List[str], operator.add] # Append logic
    final_response: str

# ==========================================
# 3. Define Model
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ==========================================
# 4. Define Nodes
# ==========================================

def planner_node(state: PlannerState):
    print("\n📋 PLANNER: Creating/Updating Plan...")
    objective = state["objective"]
    past_steps = state.get("past_steps", [])
    
    prompt = f"""
    Objective: {objective}
    
    History of steps executed so far:
    {past_steps}
    
    Create a list of remaining steps to achieve the objective.
    Return ONLY a python list of strings, e.g., ["step 1", "step 2"].
    If the objective is met, return an empty list [].
    """
    
    # For robust parsing, we should use structured output or strict parsing.
    # Here strictly prompting for a list string representation for simplicity in demo.
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # Basic cleanup to parse list
    try:
        if "[]" in content or "empty" in content.lower():
            plan = []
        else:
            # Dangerous eval for demo, use with caution or regex
            # Assuming model returns strictly ["a", "b"]
            import ast
            # cleaning markdown code blocks
            if "```" in content:
                content = content.split("```")[1].replace("python", "").replace("json", "").strip()
            plan = ast.literal_eval(content)
    except:
        print(f"    Warning: Could not parse plan: {content}. Stopping.")
        plan = []

    print(f"    Remaining Plan: {plan}")
    return {"plan": plan}

def executor_node(state: PlannerState):
    print("\n🛠️ EXECUTOR: Executing next step...")
    plan = state["plan"]
    if not plan:
        return {} # Should not happen if routed correctly
        
    current_step = plan[0]
    print(f"    Step: {current_step}")
    
    # In a real planner, the agent would decide WHICH tool to use for the step.
    # Here, we simulate execution by asking the LLM to 'perform' it or use a tool.
    
    execution_prompt = f"Perform this step: {current_step}. Provide the result."
    response = llm.invoke(execution_prompt)
    result = response.content
    
    print(f"    Result: {result[:50]}...")
    
    return {
        "past_steps": [f"Step: {current_step} -> Result: {result}"]
    }

def response_node(state: PlannerState):
    print("\n✅ FINALIZER: Generating final response...")
    prompt = f"""
    Objective: {state['objective']}
    History: {state['past_steps']}
    
    Provide a final comprehensive answer based on the history.
    """
    response = llm.invoke(prompt)
    print(f"\n💡 FINAL ANSWER:\n{response.content}")
    return {"final_response": response.content}

# ==========================================
# 5. Define Routing
# ==========================================
def router(state: PlannerState):
    if not state["plan"]:
        return "response"
    return "executor"

# ==========================================
# 6. Build Graph
# ==========================================
workflow = StateGraph(PlannerState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("response", response_node)

workflow.add_edge(START, "planner")

workflow.add_conditional_edges(
    "planner",
    router,
    {
        "executor": "executor",
        "response": "response"
    }
)

# After execution, go back to planner to update remaining steps
workflow.add_edge("executor", "planner")
workflow.add_edge("response", END)

app = workflow.compile()

# ==========================================
# 7. Run Logic
# ==========================================
if __name__ == "__main__":
    print("🤖 LangGraph Planner Agent Demo")
    print("---------------------------------")
    
    objective = "Find the population of France, assume it increases by 2% next year, and tell me the projected number."
    
    initial_state = {
        "objective": objective,
        "plan": [],
        "past_steps": []
    }
    
    # Recursion limit is high for looped agents
    config = {"recursion_limit": 10}
    
    for event in app.stream(initial_state, config=config):
        pass

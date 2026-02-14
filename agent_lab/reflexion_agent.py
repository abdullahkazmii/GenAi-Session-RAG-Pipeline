import os
import getpass
from dotenv import load_dotenv
from typing import List, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START

# Load environment variables
load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ==========================================
# 1. Define State
# ==========================================
class ReflexionState(TypedDict):
    messages: List[BaseMessage]
    critique: str
    iterations: int

# ==========================================
# 2. Define Model
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ==========================================
# 3. Define Prompts
# ==========================================
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert writer. Your goal is to write a high-quality, comprehensive answer to the user's question."),
    MessagesPlaceholder(variable_name="messages"),
    ("user", "Critique from previous iteration (if any): {critique}. \nPlease provide your updated (or initial) answer.")
])

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a harsh critic. You review the user's answer and identify missing information, logical flaws, or areas for improvement."),
    MessagesPlaceholder(variable_name="messages"),
    ("user", "Review the last answer provided by the assistant above. Provide a brief critique and suggestions for improvement. If the answer is perfect, say 'PERFECT'.")
])

# Chains
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

# ==========================================
# 4. Define Nodes
# ==========================================
def generation_node(state: ReflexionState):
    print(f"\n📝 GENERATOR (Iteration {state['iterations'] + 1})")
    request = state["messages"][0] # Original Question
    
    # We pass the history context
    response = generate_chain.invoke({
        "messages": state["messages"], 
        "critique": state.get("critique", "None")
    })
    
    print(f"   Draft: {response.content[:100]}...")
    
    # Return updated messages (append response) and increment iteration
    return {
        "messages": state["messages"] + [response],
        "iterations": state["iterations"] + 1
    }

def reflection_node(state: ReflexionState):
    print(f"\n🤔 REFLECTOR")
    # State messages now contains: [Using Question, ... , Latest Draft]
    response = reflect_chain.invoke({"messages": state["messages"]})
    
    critique = response.content
    print(f"   Critique: {critique}")
    
    return {"critique": critique}

# ==========================================
# 5. Define Routing
# ==========================================
def should_continue(state: ReflexionState):
    if state["iterations"] > 2:
        print("\n🛑 Max iterations reached. Stopping.")
        return END
    
    if "PERFECT" in state["critique"]:
        print("\n✨ Critique is 'PERFECT'. Stopping.")
        return END
        
    return "reflect" # Go to reflector

def should_loop(state: ReflexionState):
    # After reflection, always go back to generate (unless we logic-ed out in should_continue, but that's on the edge from generate)
    # Actually, standard flow: Generate -> Reflect -> Check -> (Generate or End)
    # Let's adjust: Generate -> Check -> (Reflect -> Generate) or End
    # But usually we check AFTER reflection because reflection decides if it's good.
    
    # Let's use simpler graph: Generate -> Reflect -> Condition(End or Generate)
    if "PERFECT" in state["critique"]:
        return END
    return "generate"

# ==========================================
# 6. Build Graph
# ==========================================
workflow = StateGraph(ReflexionState)

workflow.add_node("generate", generation_node)
workflow.add_node("reflect", reflection_node)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
    "reflect",
    should_loop,
    {
        END: END,
        "generate": "generate"
    }
)

app = workflow.compile()

# ==========================================
# 7. Run Logic
# ==========================================
if __name__ == "__main__":
    print("🤖 LangGraph Reflexion Agent Demo")
    print("---------------------------------")
    
    question = "Write a short poem about the Rust programming language."
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "critique": "",
        "iterations": 0
    }
    
    # Since we have a loop, we need to handle the output stream carefully
    # We'll just run it and let the nodes print
    for event in app.stream(initial_state):
        pass
        
    print("\n✅ Process Completed.")

import os
import getpass
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Ensure API key is set
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ==========================================
# 1. Define State
# ==========================================
class AgentState(TypedDict):
    # The 'add_messages' annotation ensures new messages are appended to history
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Track who is the next speaker
    next_speaker: str

# ==========================================
# 2. Define Tools (Researcher capabilities)
# ==========================================
@tool
def search_web(query: str) -> str:
    """Mock web search tool for demo purposes."""
    print(f"    [Tool] Searching web for: {query}")
    mock_results = {
        "agentic ai": "Agentic AI refers to AI systems that can independently plan, execute tasks, and use tools to achieve goals.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain.",
        "gemini": "Gemini is Google's advanced multimodal AI model."
    }
    # Return mock result or a generic fallback
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return "No specific information found, but it sounds like an interesting tech topic."

tools = [search_web]
tool_node = ToolNode(tools)

# ==========================================
# 3. Define Agents via LLM
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Bind tools to researcher LLM
researcher_model = llm.bind_tools(tools)

# Writer just needs base LLM
writer_model = llm

# ==========================================
# 4. Define Nodes (Agents)
# ==========================================

def researcher_node(state: AgentState):
    print("--- Researcher Agent ---")
    messages = state["messages"]
    # Force researcher persona
    system_message = {"role": "system", "content": "You are a Researcher. Search for information requested by the user. If you have found the information, formulate a short answer."}
    
    # Check if we should add system message (only if not present)
    # Ideally LangGraph manages this, but for simplicity we imply context
    
    response = researcher_model.invoke(messages)
    return {"messages": [response], "next_speaker": "Writer"}

def writer_node(state: AgentState):
    print("--- Writer Agent ---")
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message was from the researcher (tool output or answer), verify if it's substantial
    # Writer summarizes or polishes the researcher's output
    prompt = f"You are a Senior Writer. Review the information provided by the Researcher: '{last_message.content}'. Create a polished, blog-post style summary."
    
    response = writer_model.invoke([HumanMessage(content=prompt)])
    return {"messages": [response], "next_speaker": "FINISH"}

# ==========================================
# 5. Define Routing Logic
# ==========================================
def should_continue(state: AgentState) -> Literal["tools", "writer", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If researcher made a tool call, go to tools
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    
    # If researcher finished (no tool call), hand off to writer
    if state["next_speaker"] == "Writer":
        return "writer"
        
    return "__end__"

# ==========================================
# 6. Build Graph
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("tools", tool_node)
workflow.add_node("writer", writer_node)

# Start with Researcher
workflow.add_edge(START, "researcher")

# Conditional edge from Researcher
workflow.add_conditional_edges(
    "researcher",
    should_continue,
    {
        "tools": "tools",
        "writer": "writer",
        "__end__": END
    }
)

# Tools always go back to Researcher
workflow.add_edge("tools", "researcher")

# Writer always ends
workflow.add_edge("writer", END)

app = workflow.compile()

# ==========================================
# 7. Run Logic
# ==========================================
def run_multi_agent_demo(topic: str):
    print(f"\n🚀 Starting Multi-Agent Task: Research '{topic}'\n")
    initial_state = {"messages": [HumanMessage(content=f"Research information about {topic}")]}
    
    for event in app.stream(initial_state):
        for node_name, values in event.items():
            if "messages" in values:
                last_message = values["messages"][-1]
                
                # Print header based on node
                if node_name == "researcher":
                    print(f"\n🕵️‍♂️ RESEARCHER:")
                    # Check if it was a tool call or final answer
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                         for tool_call in last_message.tool_calls:
                            print(f"   Using Tool: {tool_call['name']} args={tool_call['args']}")
                    else:
                        print(f"   {last_message.content}")

                elif node_name == "writer":
                    print(f"\n✍️ WRITER:")
                    print(f"   {last_message.content}")
                
            print("-" * 50)
        
    print("\n✅ Workflow Completed.\n")

if __name__ == "__main__":
    print("🤖 LangGraph Multi-Agent Demo (Researcher -> Writer)")
    print("---------------------------------------------------")
    
    run_multi_agent_demo("AI Engineering")
    run_multi_agent_demo("Data Structures & Algorithms")

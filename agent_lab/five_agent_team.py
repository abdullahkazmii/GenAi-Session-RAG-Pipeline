import os
import getpass
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict
import operator

from langchain_core.messages import BaseMessage, HumanMessage
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
class StoryState(TypedDict):
    topic: str
    outline: str
    characters: str
    world_setting: str
    draft: str
    final_piece: str

# ==========================================
# 2. Define Model
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ==========================================
# 3. Define 5 Agents (Nodes)
# ==========================================

def director_node(state: StoryState):
    print("\n🎬 DIRECTOR: Creating concept and outline...")
    response = llm.invoke(f"Create a short story outline about: {state['topic']}. Provide a title and 3 bullet points for the plot.")
    print(f"\n--- [OUTLINE] ---\n{response.content}\n-----------------")
    return {"outline": response.content}

def character_designer_node(state: StoryState):
    print("\n👤 CHARACTER DESIGNER: Creating characters...")
    response = llm.invoke(f"Based on this outline, create 2 main characters with names and brief personalities:\n{state['outline']}")
    print(f"\n--- [CHARACTERS] ---\n{response.content}\n--------------------")
    return {"characters": response.content}

def world_builder_node(state: StoryState):
    print("\n🌍 WORLD BUILDER: Designing the setting...")
    response = llm.invoke(f"Based on this outline, describe the world/setting in 2 sentences:\n{state['outline']}")
    print(f"\n--- [SETTING] ---\n{response.content}\n-----------------")
    return {"world_setting": response.content}

def writer_node(state: StoryState):
    print("\n✍️ WRITER: Writing the story draft...")
    prompt = f"""
    Write a short story (approx 200 words) based on:
    Title & Plot: {state['outline']}
    Characters: {state['characters']}
    Setting: {state['world_setting']}
    """
    response = llm.invoke(prompt)
    print(f"\n--- [DRAFT] ---\n{response.content}\n---------------")
    return {"draft": response.content}

def editor_node(state: StoryState):
    print("\n📝 EDITOR: Polishing final piece...")
    prompt = f"""
    Review and polish the following story draft. Improve flow and fix any errors.
    Draft: {state['draft']}
    """
    response = llm.invoke(prompt)
    print(f"\n--- [POLISHED STORY] ---\n{response.content}\n------------------------")
    return {"final_piece": response.content}

# ==========================================
# 4. Build Graph
# ==========================================
workflow = StateGraph(StoryState)

# Add Nodes
workflow.add_node("director", director_node)
workflow.add_node("character_designer", character_designer_node)
workflow.add_node("world_builder", world_builder_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

# Add Edges (Linear Flow)
workflow.add_edge(START, "director")
workflow.add_edge("director", "character_designer")
workflow.add_edge("director", "world_builder") # Parallel?
# Let's run Chracter and World in parallel after Director, then Writer waits for both.
# LangGraph allows parallel branches.

# Director -> Characters
# Director -> World
# But Writer needs BOTH.
# In LangGraph, if we have parallel nodes, we need a way to join them.
# Writer needs to wait for both. We can use a joining edge or just sequential for simplicity first.
# Let's do Sequential for clear "communication" flow:
# Director -> Character -> World -> Writer -> Editor
workflow.add_edge("character_designer", "world_builder")
workflow.add_edge("world_builder", "writer")
workflow.add_edge("writer", "editor")
workflow.add_edge("editor", END)

app = workflow.compile()

# ==========================================
# 5. Run Logic
# ==========================================
if __name__ == "__main__":
    print("🤖 LangGraph 5-Agent Collaborative Team")
    print("---------------------------------------")
    
    topic = "Time Travel to Ancient Rome to Open a Pizza Shop"
    print(f"Task: {topic}")
    
    initial_state = {"topic": topic}
    
    result = app.invoke(initial_state)
    
    print("\n✅ FINAL STORY:\n")
    print(result["final_piece"])

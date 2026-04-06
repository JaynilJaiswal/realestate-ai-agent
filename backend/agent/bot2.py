# backend/agent/bot.py
from typing import Annotated
from langchain.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
# from langchain_core.tools import render_text_description
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import json

# Import our 4 robust tools
from backend.agent.tools import (
    search_real_estate, 
    calculate_mortgage, 
    search_live_market_data, 
    extract_text_from_url
)

load_dotenv()

# 1. Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]

tools = [search_real_estate, calculate_mortgage, search_live_market_data, extract_text_from_url]

# 2. Initialize the Multi-Model Setup

# Model A (The Worker): Llama 3 70B via Groq. 
# Blazing fast, free tier, natively supports tool calling.
worker_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
worker_with_tools = worker_llm.bind_tools(tools)

# Model B (The Synthesizer): Gemini 3.1 Flash Lite Preview handles the final polish.
# We give it a slight temperature bump for more natural, conversational phrasing.
synthesizer_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.7)



# 3. Define the Nodes

def assistant_node(state: State):
    """The Worker. Gathers data using tools."""
    # We can safely use SystemMessage again!
    system_prompt = SystemMessage(content="""You are an analytical Real Estate AI worker.
    Your job is to gather all necessary data to answer the user's query.
    Use tools as needed. Once you have all the information, output a brief internal summary of the facts found.
    Do NOT worry about formatting the final response for the user, just get the data.
    """)
    
    messages = [system_prompt] + state["messages"]
    
    # Let LangChain handle the native tool calling payload
    response = worker_with_tools.invoke(messages)
    return {"messages": [response]}

def synthesizer_node(state: State):
    """The Presenter. Formats the final client-facing response."""
    system_prompt = SystemMessage(content="""You are the lead Real Estate AI Advisor for a premium agency.
    Review the conversation history and the raw data gathered by your assistant worker. 
    Formulate a highly professional, polished, and comprehensive final response for the client.
    Ensure you include any links, prices, or calculations found. Do NOT call tools yourself.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = synthesizer_llm.invoke(messages)
    return {"messages": [response]}

# --- GRAPH BUILDER ---
builder = StateGraph(State)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("synthesizer", synthesizer_node)

builder.add_edge(START, "assistant")

# We restore the clean conditional edge. LangGraph automatically checks if 
# the AIMessage has tool_calls attached and routes accordingly.
builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", "__end__": "synthesizer"})

builder.add_edge("tools", "assistant")
builder.add_edge("synthesizer", END)

memory = MemorySaver()
agent_graph = builder.compile(checkpointer=memory)

# 7. The Execution Wrapper
def chat_with_agent(user_query: str, session_id: str = "default_user") -> str:
    config = {"configurable": {"thread_id": session_id}}
    
    events = agent_graph.stream(
        {"messages": [("user", user_query)]}, 
        config, 
        stream_mode="updates"
    )

    last_message_content = None
    
    for event in events:
        for node_name, node_state in event.items():
            print(f"\n--- Output from node: {node_name} ---")

            # Safely grab the message that was just generated
            latest_msg = node_state["messages"][-1]
            print(latest_msg.content or getattr(latest_msg, 'tool_calls', ''))
            
            # Keep updating our tracker. When the loop ends, this will hold the synthesizer's final output.
            last_message_content = latest_msg.content
        
    
    # Clean up multimodal block formats
    if isinstance(last_message_content, list):
        text_parts = [b["text"] for b in last_message_content if isinstance(b, dict) and "text" in b]
        return "".join(text_parts)
        
    return str(last_message_content)
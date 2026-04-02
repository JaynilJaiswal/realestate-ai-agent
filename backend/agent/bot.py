# backend/agent/bot.py
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq 
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from backend.agent.tools import (
    search_real_estate, 
    calculate_math_expression, 
    search_live_market_data, 
    extract_text_from_url
)

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

tools = [search_real_estate, calculate_math_expression, search_live_market_data, extract_text_from_url]

# --- THE SINGLE-AGENT GROQ ARCHITECTURE ---

# Llama 3.3 70B is smart enough to handle both routing AND final synthesis.
worker_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
worker_with_tools = worker_llm.bind_tools(tools)

def assistant_node(state: State):
    """The core agent. Gathers data and writes the final response."""
    system_prompt = SystemMessage(content="""You are an elite, professional Real Estate AI Advisor.
    You have access to a suite of tools to fetch market data, search databases, and do math.
    
    WORKFLOW:
    1. If you need data, call the appropriate tools. You can call multiple tools in sequence.
    2. Once you have gathered all the necessary information, formulate a highly professional, 
       direct, and polished final response for the client.
    3. Always include any specific numbers, rates, or Links you found in your final answer.
    4. Do not use overly fluffy or robotic corporate jargon. Be concise, factual, and helpful.
    """)
    
    messages = [system_prompt] + state["messages"]
    response = worker_with_tools.invoke(messages)
    return {"messages": [response]}

# --- GRAPH BUILDER ---
builder = StateGraph(State)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")

# LangGraph's built-in tools_condition automatically checks:
# - If the assistant used a tool -> go to "tools"
# - If the assistant just wrote text -> go to END
builder.add_conditional_edges("assistant", tools_condition)

# After a tool executes, ALWAYS return to the assistant so it can evaluate the new data
builder.add_edge("tools", "assistant")

memory = MemorySaver()
agent_graph = builder.compile(checkpointer=memory)

# --- EXECUTION WRAPPER ---
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
            
            latest_msg = node_state["messages"][-1]
            # Print either the text content or the tool call JSON to the terminal
            print(latest_msg.content or getattr(latest_msg, 'tool_calls', ''))
            
            last_message_content = latest_msg.content
            
    return str(last_message_content)
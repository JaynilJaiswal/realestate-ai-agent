# frontend/app.py
import os
import streamlit as st
import requests

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/query")

# Configure the Streamlit page to be wide for a better portfolio layout
st.set_page_config(
    page_title="AI Real Estate Agent | MLOps Portfolio",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: THE ENGINEERING PORTFOLIO ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/source-code.png", width=60)
    st.title("System Architecture")
    st.markdown("This application is a demonstration of enterprise-grade LLMOps, built with a serverless, multi-agent architecture.")
    
    st.markdown("### 🧠 AI Orchestration")
    st.markdown("""
    * **Framework:** LangGraph (Stateful Multi-Agent Routing)
    * **Inference Engine:** Groq (Ultra-low latency)
    * **Core Model:** Llama-3.3-70B-Versatile
    """)
    
    st.markdown("### 🗄️ RAG & Vector Database")
    st.markdown("""
    * **Database:** Qdrant Cloud (Managed Vector DB)
    * **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
    * **Ingestion:** Semantic chunking of 450+ real estate listings
    """)
    
    st.markdown("### 🛠️ Dynamic Tool Calling")
    st.markdown("""
    * `search_real_estate`: Semantic vector search
    * `search_live_market_data`: DuckDuckGo real-time API
    * `extract_text_from_url`: Custom semantic HTML scraper
    * `calculate_math_expression`: Sandboxed Python evaluator
    """)

    st.markdown("### 📊 Observability & LLMOps")
    st.markdown("""
    * **Telemetry:** Custom asynchronous Python interceptors
    * **Data Warehouse:** Google BigQuery (Zero-blocking inserts)
    * **Visualization:** Grafana (Token velocity, P90 Latency, Tool success rates)
    """)
    
    st.markdown("### ☁️ Infrastructure")
    st.markdown("""
    * **Compute:** Google Cloud Run (Serverless, scales to zero)
    * **Containerization:** Docker (Pre-baked model weights)
    * **API:** FastAPI
    """)
    
    st.divider()
    st.markdown("**Built by Jaynil** | [Upwork Profile](https://www.upwork.com/freelancers/~01846dade0acfca305?mp_source=share) | [GitHub](https://github.com/JaynilJaiswal/realestate-ai-agent)")


# --- MAIN CHAT INTERFACE ---
st.title("🏘️ Autonomous Real Estate AI Agent")
st.markdown("""
Welcome! I am an AI agent designed to help you find investment properties, analyze current market conditions, and calculate financial projections. 
Unlike standard chatbots, I can autonomously search a live property database, browse the internet for current mortgage rates, and execute complex math.
""")

# Quick Start Options for Clients
with st.expander("💡 Click here for sample queries to test the Agent's capabilities", expanded=False):
    st.markdown("""
    Copy and paste one of these stress tests to see the agent chain multiple tools together:
    * **Test RAG + Math:** *"Find me a highly-rated 2 bedroom apartment under $150 a night. Assuming I bought it for $300k with 20% down at 6.5% for 30 years, calculate my mortgage."*
    * **Test Web Scraping + Math:** *"Search the web for today's average 30-year fixed mortgage rate, then calculate a 30-year mortgage on a $400k home using that exact rate with 20% down."*
    * **Test Semantic Search:** *"Show me the most expensive luxury rental available in the database."*
    """)

st.divider()

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about properties, market rates, or mortgage calculations..."):
    
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Send request to the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning, calling tools, and gathering data..."):
            try:
                # We generate a pseudo-random session ID for this browser tab to track in BigQuery
                if "session_id" not in st.session_state:
                    import uuid
                    st.session_state.session_id = f"web_user_{str(uuid.uuid4())[:6]}"
                
                payload = {
                    "user_input": prompt,
                    "session_id": st.session_state.session_id
                }
                
                response = requests.post(API_URL, json=payload, timeout=60)
                response.raise_for_status() 
                
                agent_answer = response.json().get("answer", "Error: No answer returned.")
                st.markdown(agent_answer)
                
                st.session_state.messages.append({"role": "assistant", "content": agent_answer})
                
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend. Is the FastAPI server running?")
            except requests.exceptions.ReadTimeout:
                st.error("⏳ The agent took too long to respond. The system might be experiencing a cold start.")
            except Exception as e:
                st.error(f"🚨 An error occurred: {str(e)}")
# frontend/app.py
import os

import streamlit as st
import requests

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/query")

# Configure the Streamlit page
st.set_page_config(
    page_title="Real Estate AI Advisor",
    page_icon="🏘️",
    layout="centered"
)

st.title("🏘️ AI Real Estate Advisor")
st.markdown("""
Welcome to your personal Real Estate Investment Agent. 
Ask me to find properties matching your criteria, or ask me to calculate mortgage payments!
""")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., Find me a 2-bed apartment under $100/night, and calculate the mortgage for a $300k home."):
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Send request to the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing market data..."):
            try:
                # Make the POST request to our FastAPI backend
                response = requests.post(API_URL, json={"user_input": prompt})
                response.raise_for_status() # Raise an error for bad status codes
                
                # Extract the answer from the JSON response
                agent_answer = response.json().get("answer", "Error: No answer returned.")
                
                # Display the response
                st.markdown(agent_answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": agent_answer})
                
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"🚨 An error occurred: {str(e)}")
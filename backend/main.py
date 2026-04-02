from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from backend.agent.bot import chat_with_agent # Import the agent

# Load environment variables (e.g., OPENAI_API_KEY, DB credentials)
load_dotenv()

app = FastAPI(
    title="Real Estate AI Agent API",
    description="An agentic backend for automated real estate valuation and query resolution.",
    version="1.0.0"
)

# Pydantic models for strict type checking on inputs/outputs
class QueryRequest(BaseModel):
    user_input: str
    session_id: str = "default_session"

class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []

@app.get("/health")
async def health_check():
    """Simple health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "Real Estate Agent API"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main endpoint for the frontend to communicate with the Agent.
    Later, we will inject the LangGraph agent execution here.
    """
    try:
        # Pass the web request directly to our LangChain Agent
        answer = chat_with_agent(request.user_input, request.session_id)
        
        return QueryResponse(answer=answer, sources=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server with auto-reload for local development
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
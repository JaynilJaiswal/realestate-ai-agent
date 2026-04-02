# backend/agent/tools.py
from langchain.tools import tool
from langchain_qdrant import QdrantVectorStore 
from langchain_huggingface import HuggingFaceEmbeddings
from backend.database.qdrant import get_client, COLLECTION_NAME
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import math

@tool
def search_live_market_data(query: str) -> str:
    """
    Useful for finding current, real-time information that is not in the database.
    Use this for checking today's mortgage rates, current housing market trends, or neighborhood news.
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)
        
        if not results:
            return "No live data found for this query."
            
        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(f"Source {i+1}: {res['title']}\nSummary: {res['body']}\nLink: {res['href']}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error fetching live market data: {str(e)}"

@tool
def extract_text_from_url(url: str) -> str:
    """
    Useful for scraping targeted content from a specific website or article.
    Use this when the user provides a URL or when you need to read an article found via search.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # OPTIMIZATION: Instead of dumping the whole page, we only extract semantic text blocks.
        # This strips out 90% of website bloat like navbars, footers, sidebars, and raw JS/CSS.
        content_tags = soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])
        
        text = ' '.join([tag.get_text(strip=True) for tag in content_tags])
        text = ' '.join(text.split()) # Clean up excess whitespace
        
        # We can safely allow a few more characters now since the data is incredibly dense and relevant
        if len(text) > 4000:
            return text[:4000] + "\n...[Content truncated]"
        return text
        
    except Exception as e:
        return f"Failed to extract content from {url}. Error: {str(e)}"

@tool
def search_real_estate(query: str) -> str:
    """
    Useful for finding real estate properties, listings, and neighborhood information.
    Input should be a description of what the user is looking for.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        qdrant = QdrantVectorStore(
            client=get_client(),
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        
        docs = qdrant.similarity_search(query, k=3)
        
        if not docs:
            return "No matching properties found in the database. I should ask the user to adjust their criteria."
            
        print(f"Found {len(docs)} relevant listings for query: '{query}'")
        
        results = []
        for i, d in enumerate(docs):
            res = f"Listing {i+1}:\nDetails: {d.page_content}\nLink: {d.metadata.get('listing_url', 'N/A')}"
            results.append(res)
            
        return "\n\n".join(results)
    except Exception as e:
        print(f"Database error: {str(e)}")
        return "The property database is currently undergoing maintenance. Please inform the user."

@tool
def calculate_math_expression(expression: str) -> str:
    """
    Evaluates a generic mathematical expression safely. 
    Useful for ANY financial calculation: Mortgages, ROI, Cap Rate, Price per SqFt, Down Payments.
    Provide a valid Python mathematical string (e.g., "300000 * 0.20" or "250000 * (0.065/12) / (1 - (1 + 0.065/12)**-360)").
    """
    try:
        # Create a safe execution environment restricted to basic math functions
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        # Disable builtins to prevent arbitrary code execution via eval()
        result = eval(expression, {"__builtins__": None}, allowed_names)
        
        # Format the result nicely
        if isinstance(result, float):
            return f"Result: {result:.2f}"
        return f"Result: {result}"
        
    except Exception as e:
        # If the LLM writes a bad formula, this tells it to try again
        return f"Math Evaluation Error: {str(e)}. Please review and correct your formula string."
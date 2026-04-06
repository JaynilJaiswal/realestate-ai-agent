# backend/database/qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

# Grab the URL from the environment (Docker will provide this)
DB_PATH = os.path.join(os.path.dirname(__file__), "local_qdrant")
# client = QdrantClient(path=DB_PATH)
_client = None

COLLECTION_NAME = "real_estate_listings"
# OpenAI's standard text-embedding-3-small uses 1536 dimensions. 
# Adjust this if you are using a local HuggingFace embedding model.
VECTOR_SIZE = 384 

def init_db():
    """Ensures the Qdrant collection exists before we try to insert data."""

    client = get_client()

    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"Collection {COLLECTION_NAME} already exists.")

def get_client() -> QdrantClient:
    """Lazy initialization: Connects to Docker URL if present, otherwise local file."""
    global _client
    if _client is None:
        # Fetch the URL directly inside the function to ensure it catches 
        # any variables set by load_dotenv() or the terminal right before execution.
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")    
        
        if qdrant_url and qdrant_api_key:
            # Connect to Managed Qdrant Cloud
            _client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        elif qdrant_url:
            print(f"Connecting to Qdrant network service at: {qdrant_url}")
            _client = QdrantClient(url=qdrant_url)
        else:
            print(f"Connecting to local Qdrant file at: {DB_PATH}")
            _client = QdrantClient(path=DB_PATH)
            
    return _client
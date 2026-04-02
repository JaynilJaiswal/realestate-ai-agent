# test_qdrant.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from backend.database.qdrant import COLLECTION_NAME

# Force the script to look at the exposed Docker port for local testing
os.environ["QDRANT_URL"] = "http://localhost:6333"
load_dotenv()

def test_database():
    print("=== TESTING QDRANT CONNECTION ===")
    
    # 1. Test the raw connection and collection stats
    try:
        client = QdrantClient(url="http://localhost:6333")
        collection_info = client.get_collection(COLLECTION_NAME)
        print("✅ Successfully connected to Qdrant!")
        print(f"📊 Collection '{COLLECTION_NAME}' status: {collection_info.status}")
        print(f"🔢 Total vectors stored: {collection_info.points_count}")
        
        if collection_info.points_count == 0:
            print("\n🚨 ERROR: The database is connected, but it is EMPTY.")
            print("You need to run the ingestion script again.")
            return
            
    except Exception as e:
        print(f"\n🚨 ERROR: Could not connect to Qdrant at localhost:6333.")
        print(f"Details: {str(e)}")
        print("Is the Docker container running? Check with 'docker ps'.")
        return

    # 2. Test the LangChain search wrapper
    print("\n=== TESTING SEMANTIC SEARCH ===")
    try:
        print("Loading embedding model (this might take a second)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        
        test_query = "A cheap place to stay"
        print(f"Searching for: '{test_query}'...")
        
        docs = qdrant.similarity_search(test_query, k=1)
        
        if docs:
            print("✅ Search successful! Found a match:")
            print(f"Content Preview: {docs[0].page_content[:150]}...")
            print(f"Metadata: {docs[0].metadata}")
        else:
            print("⚠️ Search executed, but returned no results. Check your embeddings.")
            
    except Exception as e:
        print(f"\n🚨 ERROR: Semantic search failed.")
        print(f"Details: {str(e)}")

if __name__ == "__main__":
    test_database()
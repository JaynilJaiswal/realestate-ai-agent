# backend/scraper/ingest.py
import pandas as pd
from langchain_qdrant import QdrantVectorStore  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from backend.database.qdrant import get_client, init_db, COLLECTION_NAME
import os
from dotenv import load_dotenv
import time

load_dotenv()

def clean_price(price_str):
    """Converts price strings like '$70.00' to float 70.0"""
    if pd.isna(price_str):
        return 0.0
    if isinstance(price_str, str):
        price_str = price_str.replace('$', '').replace(',', '')
    try:
        return float(price_str)
    except ValueError:
        return 0.0

def process_and_load_data(csv_path: str):
    print(f"Starting ingestion for {csv_path}...")
    init_db()
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return

    df['price_clean'] = df['price'].apply(clean_price)
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').fillna(0)
    df['accommodates'] = pd.to_numeric(df['accommodates'], errors='coerce').fillna(0)
    df['review_scores_rating'] = pd.to_numeric(df['review_scores_rating'], errors='coerce').fillna(0)
    
    text_columns = ['name', 'description', 'neighborhood_overview', 'amenities', 'neighbourhood_cleansed', 'room_type', 'property_type', 'bathrooms_text']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Not specified")

    documents = []
    metadata = []
    
    for _, row in df.iterrows():
        description = (
            f"Property Name: {row.get('name')}. "
            f"Type: {row.get('room_type')} ({row.get('property_type')}). "
            f"Location: Located in the {row.get('neighbourhood_cleansed')} neighborhood. "
            f"Capacity & Layout: Accommodates {row.get('accommodates')} guests, "
            f"features {row.get('bedrooms')} bedrooms, and has {row.get('bathrooms_text')}. "
            f"Price: ${row.get('price_clean')} per night. "
            f"Amenities: {row.get('amenities')}. "
            f"About the property: {row.get('description')} "
            f"Neighborhood details: {row.get('neighborhood_overview')}"
        )
        documents.append(description)
        
        meta = {
            "listing_id": str(row.get("id", "")),
            "price": float(row.get("price_clean", 0)),
            "bedrooms": float(row.get("bedrooms", 0)),
            "accommodates": float(row.get("accommodates", 0)),
            "rating": float(row.get("review_scores_rating", 0)),
            "neighborhood": str(row.get("neighbourhood_cleansed", "")),
            "room_type": str(row.get("room_type", "")),
            "listing_url": str(row.get("listing_url", ""))
        }
        metadata.append(meta)

    print(f"Generating embeddings for {len(documents)} listings...")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Updated to the modern QdrantVectorStore class
    qdrant = QdrantVectorStore(
        client=get_client(),
        collection_name=COLLECTION_NAME,
        embedding=embeddings  # Singular parameter
    )
    
    batch_size = 25
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]

        try:
            qdrant.add_texts(texts=batch_docs, metadatas=batch_meta)
            print(f"Upserted batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")
            time.sleep(2)

        except Exception as e:
            print(f"Rate limit hit or error on batch {i // batch_size + 1}. Error: {e}")
            print("Backing off for 15 seconds...")
            time.sleep(15)
            qdrant.add_texts(texts=batch_docs, metadatas=batch_meta)
        
    print("Ingestion complete! Vector database is updated.")

if __name__ == "__main__":
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../listings.csv"))
    process_and_load_data(csv_file)
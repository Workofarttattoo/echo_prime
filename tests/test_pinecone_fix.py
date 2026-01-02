import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")
print(f"Testing with Token: {token[:5]}...")

try:
    print("Attempting to load all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=token)
    print("Model loaded successfully!")
    
    from pinecone import Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    print(f"Testing Pinecone Key: {api_key[:10]}...")
    pc = Pinecone(api_key=api_key)
    idxs = pc.list_indexes()
    print(f"Pinecone Indexes: {[i.name for i in idxs]}")
    
except Exception as e:
    print(f"FAILURE: {e}")

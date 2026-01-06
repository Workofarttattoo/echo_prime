import os
import uuid
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from mcp_server.registry import ToolRegistry

# Load environment variables from .env
load_dotenv()

class PineconeEngine:
    """
    Internal engine for Pinecone operations.
    """
    def __init__(self, index_name: str = "echo-prime-kb"):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = index_name
        
        # Load embedding model (local)
        # Using a small, fast model for the RAG loop
        try:
            hf_token = os.getenv("HF_TOKEN")
            candidates = []
            local_path = os.getenv("ECH0_EMBED_PATH")
            if local_path:
                candidates.append(os.path.expanduser(local_path))
            # Named model fallback
            candidates.append(os.getenv("ECH0_EMBED_MODEL", "ECH0"))

            model_id = None
            for cand in candidates:
                if cand and os.path.exists(cand):
                    model_id = cand
                    break
            if model_id is None:
                model_id = candidates[-1]
            try:
                # Try with token first (if provided), otherwise explicitly disable to avoid expired system tokens
                # Using 'token' instead of deprecated 'use_auth_token'
                self.model = SentenceTransformer(model_id, token=hf_token if hf_token else False)
            except Exception as e:
                # Fallback: explicitly disable token if it's expired or invalid
                print(f"PINECONE: HF_TOKEN failed ({e}), attempting load with token disabled...")
                self.model = SentenceTransformer(model_id, token=False)
        except Exception as e:
            try:
                # Fallback to default model if custom/local fails
                fallback_id = 'sentence-transformers/all-MiniLM-L6-v2'
                print(f"PINECONE: Falling back to {fallback_id} due to error: {e}")
                self.model = SentenceTransformer(fallback_id, token=False)
                model_id = fallback_id
            except Exception as e2:
                print(f"PINECONE ERROR: Could not load embedding model: {e2}")
                self.model_available = False
            else:
                self.model_available = True
        else:
            self.model_available = True
            
        if self.api_key and self.model_available:
            try:
                self.pc = Pinecone(api_key=self.api_key)
                # Create index if not exists (Serverless)
                existing_indexes = [idx.name for idx in self.pc.list_indexes()]
                if self.index_name not in existing_indexes:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=384, 
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region=self.environment
                        )
                    )
                self.index = self.pc.Index(self.index_name)
                self.online = True
            except Exception as e:
                print(f"PINECONE ERROR: Initialization failed: {e}")
                self.online = False
        else:
            self.pc = None
            self.index = None
            self.online = False
            if not self.api_key:
                print("PINECONE: API Key missing, running in offline mode.")

    def upsert(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if not self.online: return "Pinecone is currently offline."
        vector = self.model.encode(text).tolist()
        id_ = str(uuid.uuid4())
        meta = metadata or {}
        meta["text"] = text
        self.index.upsert(vectors=[(id_, vector, meta)])
        return f"Stored in long-term memory (id: {id_})"

    def search(self, query: str, top_k: int = 5) -> str:
        if not self.online: return "Pinecone is currently offline."
        query_vector = self.model.encode(query).tolist()
        results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        if not results.matches: return "No relevant deep memory found."
        
        output = []
        for match in results.matches:
            text = match.metadata.get("text", "N/A")
            output.append(f"- {text} (score: {match.score:.2f})")
        return "Deep Memory Recall:\n" + "\n".join(output)

# Initialize global engine
_engine = PineconeEngine()

@ToolRegistry.register(name="pinecone_store")
def pinecone_store(text: str) -> str:
    """Stores a fact or observation into ECH0's long-term Pinecone memory."""
    return _engine.upsert(text)

@ToolRegistry.register(name="pinecone_search")
def pinecone_search(query: str) -> str:
    """Searches ECH0's long-term Pinecone memory for relevant background info."""
    return _engine.search(query)

def get_pinecone_engine():
    return _engine

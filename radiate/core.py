import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables
load_dotenv()


class Radiate:
    """
    Main Radiate class for RAG operations.
    
    Example:
        radiate = Radiate()
        await radiate.ingest('./docs')
        answer = await radiate.query('How do I authenticate?')
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "radiate_docs"
    ):
        """
        Initialize Radiate with API credentials.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            qdrant_url: Qdrant cluster URL (defaults to QDRANT_URL env var)
            qdrant_api_key: Qdrant API key (defaults to QDRANT_API_KEY env var)
            collection_name: Name of Qdrant collection to use
        """
        # Load API keys (prioritize args, fallback to env vars)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        
        # Validate required credentials
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env or pass as argument.")
        if not self.qdrant_url:
            raise ValueError("Qdrant URL not found. Set QDRANT_URL in .env or pass as argument.")
        if not self.qdrant_api_key:
            raise ValueError("Qdrant API key not found. Set QDRANT_API_KEY in .env or pass as argument.")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        
        print(f"Radiate initialized successfully!")
        print(f"Collection: {self.collection_name}")
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # OpenAI text-embedding-3-small produces 1536-dimensional vectors
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimensions
                    distance=Distance.COSINE  # Use cosine similarity
                )
            )
            print(f"Created new collection: {self.collection_name}")
        else:
            print(f"Using existing collection: {self.collection_name}")
    
    def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

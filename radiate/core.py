import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import Optional, List, Dict, Any


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
        collection_name: str = "radiate_docs",
        validate_connections: bool = False
    ):
        """
        Initialize Radiate with API credentials.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            qdrant_url: Qdrant cluster URL (defaults to QDRANT_URL env var)
            qdrant_api_key: Qdrant API key (defaults to QDRANT_API_KEY env var)
            collection_name: Name of Qdrant collection to use
            validate_connections: Whether to validate API connections on initialization
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env or pass as argument. "
                "Get your key at: https://platform.openai.com/api-keys"
            )
        if not self.qdrant_url:
            raise ValueError("Qdrant URL not found. Set QDRANT_URL in .env or pass as argument.")
        if not self.qdrant_api_key:
            raise ValueError("Qdrant API key not found. Set QDRANT_API_KEY in .env or pass as argument.")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        
        if validate_connections:
            self._validate_openai_key()
        
        self._ensure_collection_exists()

    
    def _validate_openai_key(self):
        """Validate OpenAI API key by making a test request."""
        try:
            self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "invalid_api_key" in error_str:
                raise ValueError(
                    "Invalid OpenAI API key. Get your key at: https://platform.openai.com/api-keys"
                )
            else:
                raise ValueError(f"OpenAI connection failed: {error_str}")
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            error_str = str(e).lower()
            if "nodename nor servname" in error_str or "connecterror" in error_str:
                raise ValueError(
                    "Unable to connect to Qdrant. Check QDRANT_URL in .env file. "
                    "Format should be: https://your-cluster.cloud.qdrant.io:6333"
                ) from None
            elif "unauthorized" in error_str or "403" in error_str or "401" in error_str:
                raise ValueError(
                    "Invalid Qdrant API key. Check QDRANT_API_KEY in .env file. "
                    "Get your key from: https://cloud.qdrant.io"
                ) from None
            else:
                raise ValueError(f"Qdrant connection failed: {str(e)}") from None

    def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            ValueError: If OpenAI API request fails
        """
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            error_str = str(e)
            if "401" in error_str:
                raise ValueError("Invalid OpenAI API key")
            elif "429" in error_str:
                raise ValueError("OpenAI rate limit exceeded. Please retry after a moment.")
            elif "insufficient_quota" in error_str:
                raise ValueError("OpenAI account has insufficient credits. Please add funds.")
            else:
                raise ValueError(f"OpenAI embedding generation failed: {error_str}")


    def ingest(self, path: str, pattern: str = "*.txt") -> Dict[str, Any]:
        """
        Ingest documents from a file or directory.
        
        Args:
            path: Path to file or directory
            pattern: File pattern for directory ingestion
            
        Returns:
            Ingestion results
        """
        from radiate.ingest import DocumentIngester
        ingester = DocumentIngester(self)
        
        if os.path.isfile(path):
            return ingester.ingest_file(path)
        elif os.path.isdir(path):
            return ingester.ingest_directory(path, pattern=pattern)
        else:
            raise ValueError(f"Path not found: {path}")


    def query(self, question: str, top_k: int = 3) -> str:
        """
        Query ingested documents.
        
        Args:
            question: Question to ask
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Context from relevant documents
        """
        from radiate.query import QueryEngine
        engine = QueryEngine(self)
        return engine.query(question, top_k=top_k)
    

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        from radiate.query import QueryEngine
        engine = QueryEngine(self)
        return engine.search(query, top_k=top_k)

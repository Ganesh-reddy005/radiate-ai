import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Import our new flexible embedding system
from radiate.embeddings import create_embeddings, EmbeddingProvider

load_dotenv()


class Radiate:
    """
    Main Radiate class for RAG operations with flexible embedding providers.
    
    Example:
        # Local embeddings (free, no API key)
        radiate = Radiate(embedding_provider="local")
        
        # OpenAI embeddings
        radiate = Radiate(
            embedding_provider="openai",
            openai_api_key="sk-..."
        )
    """
    
    def __init__(
        self,
        # Embedding configuration
        embedding_provider: str = "local",
        embedding_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        
        # Qdrant configuration
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "radiate_docs",
        
        # Features
        track_costs: bool = True,
        validate_connections: bool = False
    ):
        """
        Initialize Radiate with flexible embedding provider.
        
        Args:
            embedding_provider: "local" (free), "openai" (paid), or "openrouter"
            embedding_model: Specific model name (optional, uses defaults)
            openai_api_key: API key for OpenAI or OpenRouter
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
            collection_name: Name of Qdrant collection
            track_costs: Enable cost tracking and caching
            validate_connections: Test connections on init
        """
        # Qdrant setup
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        
        if not self.qdrant_url:
            raise ValueError(
                "Qdrant URL not found. Set QDRANT_URL in .env or pass as argument."
            )
        if not self.qdrant_api_key:
            raise ValueError(
                "Qdrant API key not found. Set QDRANT_API_KEY in .env or pass as argument."
            )
        
        try:
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url, 
                api_key=self.qdrant_api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to connect to Qdrant: {str(e)}")
        
        # Embedding provider setup
        try:
            self.embedder: EmbeddingProvider = create_embeddings(
                provider=embedding_provider,
                model_name=embedding_model,
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                track_costs=track_costs
            )
            print(f"Radiate initialized with {embedding_provider} embeddings")
        except Exception as e:
            raise ValueError(f"Failed to initialize embedding provider: {str(e)}")
        
        # Get embedding dimension for collection
        test_vec = self.embedder.embed("test")
        self.embedding_dim = len(test_vec)
        
        # Create collection
        self._ensure_collection_exists()
        
        if validate_connections:
            self._validate_setup()
    
    def _ensure_collection_exists(self):
        """Create or validate Qdrant collection with dimension checking."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                # Collection exists - validate dimension
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                existing_dim = collection_info.config.params.vectors.size
                
                if existing_dim != self.embedding_dim:
                    raise ValueError(
                        f"\nDimension mismatch in collection '{self.collection_name}':\n"
                        f"  Existing: {existing_dim} dimensions\n"
                        f"  Current model: {self.embedding_dim} dimensions\n\n"
                        f"Solutions:\n"
                        f"  1. Delete collection:\n"
                        f"     radiate.delete_collection(confirm=True)\n"
                        f"  2. Use different collection name:\n"
                        f"     Radiate(collection_name='radiate_docs_new')\n"
                        f"  3. Switch to compatible embedding model"
                    )
                
                print(f"Using existing collection '{self.collection_name}' (dim={existing_dim})")
            else:
                # Create new collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection '{self.collection_name}' (dim={self.embedding_dim})")
        
        except ValueError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "unauthorized" in error_str or "403" in error_str or "401" in error_str:
                raise ValueError("Invalid Qdrant API key. Check QDRANT_API_KEY in .env") from None
            else:
                raise ValueError(f"Qdrant error: {str(e)}") from None
    
    def delete_collection(self, confirm: bool = False):
        """
        Delete the current collection and recreate it.
        
        WARNING: This permanently deletes all data in the collection.
        
        Args:
            confirm: Must be True to proceed (safety check)
        
        Raises:
            ValueError: If confirm is not True
        
        Example:
            radiate = Radiate(embedding_provider="local")
            radiate.delete_collection(confirm=True)
        """
        if not confirm:
            raise ValueError(
                "Collection deletion requires explicit confirmation.\n"
                "This will permanently delete all data.\n\n"
                "To proceed:\n"
                "  radiate.delete_collection(confirm=True)"
            )
        
        try:
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection '{self.collection_name}'")
            
            # Recreate with current embedding dimensions
            self._ensure_collection_exists()
        
        except Exception as e:
            raise ValueError(f"Failed to delete collection: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """
        List all available Qdrant collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.qdrant_client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            raise ValueError(f"Failed to list collections: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection metadata
        """
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_dimension": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.name,
                "points_count": info.points_count,
                "status": info.status.name
            }
        except Exception as e:
            raise ValueError(f"Failed to get collection info: {str(e)}")
    
    def _validate_setup(self):
        """Validate that everything is working."""
        try:
            # Test embedding
            vec = self.embedder.embed("validation test")
            print(f"Embedding working (dim={len(vec)})")
            
            # Test Qdrant
            collections = self.qdrant_client.get_collections()
            print(f"Qdrant connected ({len(collections.collections)} collections)")
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self.embedder.embed(text)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (faster than one-by-one).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embedder.embed_batch(texts)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embeddings and costs.
        
        Returns:
            Dictionary with stats
        """
        return self.embedder.get_stats()
    
    def ingest(self, path: str, pattern: str = "*.txt") -> Dict[str, Any]:
        """
        Ingest documents from a file or directory.
        
        Args:
            path: Path to file or directory
            pattern: File pattern for directory ingestion
            
        Returns:
            Ingestion results with cost tracking
        """
        from radiate.ingest import DocumentIngester
        ingester = DocumentIngester(self)
        
        if os.path.isfile(path):
            result = ingester.ingest_file(path)
        elif os.path.isdir(path):
            result = ingester.ingest_directory(path, pattern=pattern)
        else:
            raise ValueError(f"Path not found: {path}")
        
        # Add cost stats to result
        result["embedding_stats"] = self.get_stats()
        return result
    
    def query(self, question: str, top_k: int = 3, mode: str = "dense") -> str:
        """
        Query ingested documents.

        Args:
            question: Question to ask
            top_k: Number of relevant chunks to retrieve
            mode: Retrieval mode - "dense" (default), "sparse" (BM25), or "hybrid"

        Returns:
            Context from relevant documents

        Examples:
            # Dense vector search
            radiate.query("What is machine learning?")
            
            # Hybrid search (better accuracy)
            radiate.query("API rate limit", mode="hybrid")
        """
        from radiate.query import QueryEngine
        engine = QueryEngine(self)
        return engine.query(question, top_k=top_k, mode=mode)
    
    def search(self, query: str, top_k: int = 5, mode: str = "dense") -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            mode: Retrieval mode - "dense", "sparse", or "hybrid"
        
        Returns:
            List of search results with scores
        """
        from radiate.query import QueryEngine
        engine = QueryEngine(self)
        return engine.search(query, top_k=top_k, mode=mode)

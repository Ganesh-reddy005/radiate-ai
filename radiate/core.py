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

    #-------------check collection----------------------

    def _ensure_collection_exists(self):
        """Create or validate Qdrant collection with dimension checking and payload indexes."""
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
                # Import payload schema types
                from qdrant_client.models import PayloadSchemaType
                
                # Create new collection with payload indexes
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                
                # Create indexes for filtering
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="chunk_index",
                    field_schema=PayloadSchemaType.INTEGER
                )
                
                print(f"Created collection '{self.collection_name}' (dim={self.embedding_dim})")
                print("Created payload indexes: source (keyword), chunk_index (integer)")
        
        except ValueError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "unauthorized" in error_str or "403" in error_str or "401" in error_str:
                raise ValueError("Invalid Qdrant API key. Check QDRANT_API_KEY in .env") from None
            else:
                raise ValueError(f"Qdrant error: {str(e)}") from None

    #-------------delete collection----------------------
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
    
    #-----------------List collections -------------------------------------
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
    
    #----------------------------validte setup------------------------------------
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
    
    #-------------------------------------------------------------------------------------------------------------
    #--------------------------------embeddings-----------------------------------------

    #----------------------1.Get embeddings------------------------------------
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        return self.embedder.embed(text)
    
    #---------------------------------get_embeddings_batch------------------------
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (faster than one-by-one).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embedder.embed_batch(texts)
    
    #--------------------get_status---------------------------
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embeddings and costs.
        
        Returns:
            Dictionary with stats
        """
        return self.embedder.get_stats()
    
    #----------------------INGEST----------------------------------
    def ingest(
        self, 
        path: str, 
        pattern: str = None, 
        chunk_mode: str = 'smart',
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: Dict[str, Any] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        skip_errors: bool = False,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest documents from a file or directory.
        
        Args:
            path: File or directory path
            pattern: File pattern (None = auto-detect .txt, .md, .pdf)
            chunk_mode: 'smart' (default) or 'token'
            chunk_size: Max tokens per chunk (default: 512)
            overlap: Token overlap between chunks (default: 50)
            metadata: Custom metadata to attach to all chunks
            batch_size: Embedding batch size (default: 32)
            show_progress: Display progress bar (default: True)
            skip_errors: Continue on file errors (default: False)
            recursive: Scan subdirectories (default: False)
            
        Returns:
            Ingestion results with cost stats
            
        Example:
            result = radiate.ingest(
                "docs/",
                chunk_size=1024,
                overlap=100,
                metadata={"version": "1.0"},
                recursive=True
            )
        """
        # Validation
        if chunk_size < 50:
            raise ValueError("chunk_size must be >= 50")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        from radiate.ingest import DocumentIngester
        ingester = DocumentIngester(self)
        
        if os.path.isfile(path):
            result = ingester.ingest_file(
                path,
                metadata=metadata,
                chunk_mode=chunk_mode,
                chunk_size=chunk_size,
                overlap=overlap
            )
            # Standardize response
            result['total_chunks'] = result.get('chunks_ingested', 0)
            result['total_files'] = 1
        elif os.path.isdir(path):
            result = ingester.ingest_directory(
                path, 
                pattern=pattern,
                chunk_mode=chunk_mode,
                chunk_size=chunk_size,
                overlap=overlap,
                metadata=metadata,
                show_progress=show_progress,
                skip_errors=skip_errors,
                recursive=recursive
            )
        else:
            raise ValueError(f"Path not found: {path}")
        
        # Add cost stats
        result["embedding_stats"] = self.get_stats()
        return result


    #-------------------------Query-------------------------------
    def query(self, question: str, top_k: int = 3, mode: str = "dense") -> list:
        from radiate.query import QueryEngine
        engine = QueryEngine(self)
        result = engine.query(question, top_k=top_k, mode=mode)
        # Guarantee result is always list of dicts for LLM
        if isinstance(result, str):
            return [{"text": result}]
        elif isinstance(result, list):
            # Check for list of dicts—return as is for LLM
            if len(result) > 0 and isinstance(result[0], dict) and "text" in result[0]:
                return result
            else:
                # List of strings
                return [{"text": str(r)} for r in result]
        elif isinstance(result, dict) and "chunks" in result:
            return result["chunks"]
        else:
            return [{"text": str(result)}]

    
    #---------------------SEARCH------------------------------------
    def search(self, query: str, top_k: int = 5, mode: str = "dense") -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            mode: Retrieval mode - "dense", "sparse", or "hybrid"
        
        Returns:
            List of search results with scores
            For testing purpose
        """
        from radiate.query import QueryEngine
        engine = QueryEngine(self)
        return engine.search(query, top_k=top_k, mode=mode)


#-------------------------------------------------------------------------------------------
    #-------------------------async functionality--------------------------
    async def ingest_async(
        self, 
        path: str, 
        pattern: str = None,
        chunk_mode: str = 'smart',
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: Dict[str, Any] = None,
        max_concurrent_files: int = 3,
        batch_size: int = 32,
        show_progress: bool = True,
        skip_errors: bool = False,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Async ingest documents (10x faster for large datasets).
        
        Args:
            path: File or directory path
            pattern: File pattern (None = auto-detect)
            chunk_mode: 'smart' or 'token'
            chunk_size: Max tokens per chunk (default: 512)
            overlap: Token overlap (default: 50)
            metadata: Custom metadata
            max_concurrent_files: Max concurrent file processing
            batch_size: Embedding batch size (default: 32)
            show_progress: Display progress (default: True)
            skip_errors: Continue on errors (default: False)
            recursive: Scan subdirectories (default: False)
            
        Returns:
            Ingestion results with stats
            
        Example:
            result = await radiate.ingest_async(
                "docs/",
                chunk_size=1024,
                recursive=True
            )
        """
        # Validation
        if chunk_size < 50:
            raise ValueError("chunk_size must be >= 50")
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        from radiate.ingest_async import AsyncDocumentIngester
        ingester = AsyncDocumentIngester(self)
        
        if os.path.isfile(path):
            result = await ingester.ingest_file_async(
                path,
                metadata=metadata,
                chunk_mode=chunk_mode,
                chunk_size=chunk_size,
                overlap=overlap
            )
            # Standardize response
            result['total_chunks'] = result.get('chunks_ingested', 0)
            result['total_files'] = 1
        elif os.path.isdir(path):
            result = await ingester.ingest_directory_async(
                path, 
                pattern=pattern,
                chunk_mode=chunk_mode,
                chunk_size=chunk_size,
                overlap=overlap,
                metadata=metadata,
                max_concurrent_files=max_concurrent_files,
                show_progress=show_progress,
                skip_errors=skip_errors,
                recursive=recursive
            )
        else:
            raise ValueError(f"Path not found: {path}")
        
        result["embedding_stats"] = self.get_stats()
        return result


#--------------------getting chunks data---------------

    def get_all_chunks(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from collection with pagination.
        
        Args:
            limit: Maximum number of chunks to retrieve (default: 100)
            offset: Number of chunks to skip (default: 0)
            
        Returns:
            List of chunks with metadata
            
        Example:
            # Get first 10 chunks
            chunks = radiate.get_all_chunks(limit=10)
            
            # Get next 10 chunks
            chunks = radiate.get_all_chunks(limit=10, offset=10)
        """
        try:
            # Scroll through collection
            points, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_vectors=False,  # Don't return vectors (saves bandwidth)
                with_payload=True
            )
            
            chunks = []
            for point in points:
                chunks.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "source": point.payload.get("source", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "total_chunks": point.payload.get("total_chunks", 0),
                    "metadata": {k: v for k, v in point.payload.items() 
                            if k not in ["text", "source", "chunk_index", "total_chunks"]}
                })
            
            return chunks
        
        except Exception as e:
            raise ValueError(f"Failed to retrieve chunks: {str(e)}")
        


    #--------------------------chunks_by_source----------------------
    def get_chunks_by_source(self, source: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific source file.
        
        Args:
            source: Source file path
            limit: Maximum chunks to return
            
        Returns:
            List of chunks from that source
            
        Example:
            chunks = radiate.get_chunks_by_source("api_docs.txt")
            print(f"Found {len(chunks)} chunks from api_docs.txt")
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Filter by source
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source)
                        )
                    ]
                ),
                limit=limit,
                with_vectors=False,
                with_payload=True
            )
            
            points = results[0]
            
            chunks = []
            for point in points:
                chunks.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "source": point.payload.get("source", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "total_chunks": point.payload.get("total_chunks", 0),
                    "metadata": point.payload
                })
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x["chunk_index"])
            
            return chunks
        
        except Exception as e:
            raise ValueError(f"Failed to retrieve chunks: {str(e)}")

#----------------------get_chunk_by_id-----------------------------------
    def get_chunk_by_id(self, chunk_id: int) -> Dict[str, Any]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk data
            
        Example:
            chunk = radiate.get_chunk_by_id(123456789)
            print(chunk['text'])
        """
        try:
            points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_vectors=False,
                with_payload=True
            )
            
            if not points:
                raise ValueError(f"Chunk ID {chunk_id} not found")
            
            point = points[0]
            return {
                "id": point.id,
                "text": point.payload.get("text", ""),
                "source": point.payload.get("source", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "total_chunks": point.payload.get("total_chunks", 0),
                "metadata": point.payload
            }
        
        except Exception as e:
            raise ValueError(f"Failed to retrieve chunk: {str(e)}")


#--------------------------list_sources-----------------------
    def list_sources(self) -> List[str]:
        """
        List all unique source files in the collection.
        
        Returns:
            List of source file paths
            
        Example:
            sources = radiate.list_sources()
            print(f"Ingested files: {sources}")
        """
        try:
            # Get all chunks
            all_chunks = self.get_all_chunks(limit=10000)
            
            # Extract unique sources
            sources = list(set([chunk["source"] for chunk in all_chunks]))
            sources.sort()
            
            return sources
        
        except Exception as e:
            raise ValueError(f"Failed to list sources: {str(e)}")


#-------print_chunk_summary----------------------
    def print_chunk_summary(self, chunk: Dict[str, Any]):
        """
        Pretty print a chunk for inspection.
        
        Args:
            chunk: Chunk dictionary
            
        Example:
            chunks = radiate.get_all_chunks(limit=5)
            for chunk in chunks:
                radiate.print_chunk_summary(chunk)
        """
        print(f"\n{'='*60}")
        print(f"Chunk ID: {chunk['id']}")
        print(f"Source: {chunk['source']}")
        print(f"Chunk: {chunk['chunk_index'] + 1}/{chunk['total_chunks']}")
        print(f"{'='*60}")
        print(f"\nText Preview:")
        print(f"{chunk['text'][:200]}...")
        print(f"\nFull Length: {len(chunk['text'])} characters")

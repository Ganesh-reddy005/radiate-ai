import os
import hashlib
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

# Try importing optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, track_costs: bool = True):
        self.track_costs = track_costs
        self.total_embeddings = 0
        self.cached_embeddings = 0
        self.total_cost = 0.0
        self.cache = {}
    
    @abstractmethod
    def _embed_single(self, text: str) -> List[float]:
        """Implement this in each provider"""
        pass
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed(self, text: str) -> List[float]:
        """Embed single text with caching"""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        cache_key = self._get_cache_key(text)
        
        # Check cache
        if cache_key in self.cache:
            self.cached_embeddings += 1
            return self.cache[cache_key]
        
        # Generate embedding
        try:
            embedding = self._embed_single(text)
            self.total_embeddings += 1
            self.cache[cache_key] = embedding
            return embedding
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        return [self.embed(text) for text in texts if text.strip()]
    
    def get_stats(self) -> Dict:
        """Return statistics"""
        total = max(self.total_embeddings + self.cached_embeddings, 1)
        return {
            "total_embeddings_generated": self.total_embeddings,
            "cached_embeddings": self.cached_embeddings,
            "cache_hit_rate": f"{(self.cached_embeddings / total * 100):.1f}%",
            "total_cost": f"${self.total_cost:.4f}",
            "cost_saved": f"${(self.cached_embeddings * self._get_cost_per_embedding()):.4f}"
        }
    
    @abstractmethod
    def _get_cost_per_embedding(self) -> float:
        """Cost per embedding for this provider"""
        pass


class LocalEmbeddings(EmbeddingProvider):
    """Local embedding model using sentence-transformers (FREE)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", track_costs: bool = True):
        super().__init__(track_costs)
        
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers not installed. Install with: "
                "pip install sentence-transformers"
            )
        
        try:
            print(f"Loading local model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            print(f"Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model '{model_name}': {str(e)}")
    
    def _embed_single(self, text: str) -> List[float]:
        """Generate embedding using local model"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _get_cost_per_embedding(self) -> float:
        return 0.0  # Local models are free


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding API"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        track_costs: bool = True
    ):
        super().__init__(track_costs)
        
        if not HAS_OPENAI:
            raise ImportError("openai not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env "
                "or pass api_key parameter"
            )
        
        self.model_name = model_name
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            # Test connection
            self.client.embeddings.create(model=self.model_name, input="test")
            print(f"✅ Connected to OpenAI ({self.model_name})")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to OpenAI: {str(e)}")
        
        # Pricing (as of Nov 2024)
        self.cost_per_1k_tokens = {
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013,
            "text-embedding-ada-002": 0.00010
        }.get(model_name, 0.00002)
    
    def _embed_single(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        
        # Update cost tracking
        tokens_used = len(text.split()) * 1.3  # Rough estimate
        self.total_cost += (tokens_used / 1000) * self.cost_per_1k_tokens
        
        return response.data[0].embedding
    
    def _get_cost_per_embedding(self) -> float:
        return self.cost_per_1k_tokens * 10  # Assume ~10 tokens per embedding


class OpenRouterEmbeddings(EmbeddingProvider):
    """
    OpenRouter doesn't provide embedding models directly.
    This uses OpenRouter's LLM to generate embeddings (not recommended for production).
    For production, use LocalEmbeddings or OpenAIEmbeddings.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "meta-llama/llama-3.2-3b-instruct:free",
        track_costs: bool = True
    ):
        super().__init__(track_costs)
        
        if not HAS_OPENAI:
            raise ImportError("openai not installed. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in .env "
                "or pass api_key parameter"
            )
        
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        print(f"⚠️  Using OpenRouter LLM for embeddings (not ideal)")
        print(f"💡 Recommendation: Use LocalEmbeddings for better quality")
    
    def _embed_single(self, text: str) -> List[float]:
        """
        Generate pseudo-embedding using LLM output.
        This is a workaround - not recommended for production.
        """
        # This is a hacky approach - in reality you should use proper embedding models
        raise NotImplementedError(
            "OpenRouter doesn't support embedding models. "
            "Use LocalEmbeddings (free) or OpenAIEmbeddings instead."
        )
    
    def _get_cost_per_embedding(self) -> float:
        return 0.0  # Free tier


def create_embeddings(
    provider: str = "local",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    track_costs: bool = True
) -> EmbeddingProvider:
    """
    Factory function to create embedding provider.
    
    Args:
        provider: "local", "openai", or "openrouter"
        model_name: Model name (provider-specific)
        api_key: API key if needed
        track_costs: Enable cost tracking
    
    Returns:
        EmbeddingProvider instance
    
    Examples:
        >>> # Local model (free, no API key)
        >>> embedder = create_embeddings("local")
        
        >>> # OpenAI
        >>> embedder = create_embeddings("openai", api_key="sk-...")
        
        >>> # Custom local model
        >>> embedder = create_embeddings("local", model_name="all-mpnet-base-v2")
    """
    
    if provider == "local":
        model = model_name or "all-MiniLM-L6-v2"
        return LocalEmbeddings(model_name=model, track_costs=track_costs)
    
    elif provider == "openai":
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbeddings(api_key=api_key, model_name=model, track_costs=track_costs)
    
    elif provider == "openrouter":
        return OpenRouterEmbeddings(api_key=api_key, track_costs=track_costs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: 'local', 'openai', 'openrouter'"
        )

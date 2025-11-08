from typing import List, Dict, Any


class QueryEngine:
    """Handles semantic search queries with support for hybrid retrieval."""
    
    def __init__(self, radiate_instance):
        """
        Initialize query engine.
        
        Args:
            radiate_instance: Radiate class instance for API access
        """
        self.radiate = radiate_instance
        self._hybrid_retriever = None
    
    def _get_hybrid_retriever(self):
        """Lazy initialization of hybrid retriever."""
        if self._hybrid_retriever is None:
            from radiate.retrieval import HybridRetriever
            self._hybrid_retriever = HybridRetriever(self.radiate)
        return self._hybrid_retriever
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        mode: str = "dense"
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            mode: Retrieval mode - "dense" (default), "sparse" (BM25), or "hybrid"
        
        Returns:
            List of relevant chunks with metadata and scores
        
        Examples:
            # Dense vector search (default)
            results = engine.search("machine learning")
            
            # BM25 keyword search
            results = engine.search("API error 429", mode="sparse")
            
            # Hybrid search (best of both)
            results = engine.search("reset password", mode="hybrid")
        """
        if mode in ["sparse", "hybrid"]:
            retriever = self._get_hybrid_retriever()
            return retriever.search(query, top_k=top_k, mode=mode)
        
        else:
            # Default dense search (backward compatible)
            query_embedding = self.radiate.get_embedding(query)
            
            search_results = self.radiate.qdrant_client.search(
                collection_name=self.radiate.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            results = []
            for hit in search_results:
                results.append({
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "source": hit.payload.get("source", ""),
                    "chunk_index": hit.payload.get("chunk_index", 0),
                    "metadata": {k: v for k, v in hit.payload.items() 
                               if k not in ["text", "source", "chunk_index"]}
                })
            
            return results
    
    def query(
        self, 
        question: str, 
        top_k: int = 3,
        mode: str = "dense"
    ) -> str:
        """
        Query documents and return formatted context.
        
        Args:
            question: Question to answer
            top_k: Number of chunks to retrieve
            mode: Retrieval mode - "dense", "sparse", or "hybrid"
        
        Returns:
            Formatted context from relevant chunks
        """
        results = self.search(question, top_k=top_k, mode=mode)
        
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for r in results:
            score_label = "RRF" if mode == "hybrid" and "rrf_score" in r else "Score"
            score_value = r.get('rrf_score', r.get('score', 0))
            
            context_parts.append(
                f"[Source: {r['source']}, Chunk {r['chunk_index']}, "
                f"{score_label}: {score_value:.4f}]\n{r['text']}"
            )
        
        return "\n\n".join(context_parts)

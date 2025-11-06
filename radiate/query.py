from typing import List, Dict, Any


class QueryEngine:
    """Handles semantic search queries against ingested documents."""
    
    def __init__(self, radiate_instance):
        """
        Initialize query engine with a Radiate instance.
        
        Args:
            radiate_instance: Radiate class instance for API access
        """
        self.radiate = radiate_instance
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata and scores
        """
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
    
    def query(self, question: str, top_k: int = 3) -> str:
        """
        Query documents and return formatted context.
        
        Args:
            question: Question to answer
            top_k: Number of chunks to retrieve
            
        Returns:
            Formatted context from relevant chunks
        """
        results = self.search(question, top_k=top_k)
        
        if not results:
            return "No relevant information found."
        
        context = "\n\n".join([
            f"[Source: {r['source']}, Chunk {r['chunk_index']}, Score: {r['score']:.2f}]\n{r['text']}"
            for r in results
        ])
        
        return context

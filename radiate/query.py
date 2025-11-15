from typing import List, Dict, Any, Union

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
    
 # MODIFY search() method signature:

    def search(
        self, 
        query: str, 
        top_k: int = 5,
        mode: str = "dense",
        rerank: bool = False  # NEW
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents with optional reranking.
        
        Args:
            query: Search query text
            top_k: Number of results to return (after reranking if enabled)
            mode: Retrieval mode - "dense", "sparse", or "hybrid"
            rerank: Whether to apply cross-encoder reranking
        
        Returns:
            List of relevant chunks with metadata and scores
        """
        # Determine retrieval count: get more candidates if reranking
        retrieve_k = top_k * 3 if rerank and self.radiate.reranker else top_k
        
        # Retrieve candidates
        if mode in ["sparse", "hybrid"]:
            retriever = self._get_hybrid_retriever()
            results = retriever.search(query, top_k=retrieve_k, mode=mode)
        else:
            # Default dense search
            query_embedding = self.radiate.get_embedding(query)
            
            search_results = self.radiate.qdrant_client.search(
                collection_name=self.radiate.collection_name,
                query_vector=query_embedding,
                limit=retrieve_k
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
        
        # NEW: Apply reranking if enabled
        if rerank and self.radiate.reranker and results:
            results = self._rerank_results(query, results, top_k)
        
        return results[:top_k]


    # NEW METHOD: Add this to QueryEngine class

    def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder model.
        
        Args:
            query: Original query
            results: List of result dicts with 'text' field
            top_k: Number of top results to return
        
        Returns:
            Reranked results with updated scores
        """
        # Extract texts for reranking
        texts = [r["text"] for r in results]
        
        # Rerank and get scores
        reranked_texts, rerank_scores = self.radiate.reranker.rerank(
            query, 
            texts, 
            top_k=top_k,
            return_scores=True
        )
        
        # Map reranked texts back to original results
        text_to_result = {r["text"]: r for r in results}
        
        reranked_results = []
        for text, score in zip(reranked_texts, rerank_scores):
            result = text_to_result[text].copy()
            result["rerank_score"] = float(score)
            result["original_score"] = result.get("score", result.get("rrf_score", 0))
            reranked_results.append(result)
        
        return reranked_results


    def query(
        self,
        question: str,
        top_k: int = 3,
        mode: str = "dense",
        rerank: bool = False,
        metrics: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Query documents and return formatted context or structured results.
        
        Args:
            question: Question to answer
            top_k: Number of chunks to retrieve
            mode: Retrieval mode - "dense", "sparse", or "hybrid"
            rerank: Whether to apply reranking (requires reranker enabled)
            metrics: If True, return structured output with quality metrics
            
        Returns:
            If metrics=False: Formatted string (backward compatible)
            If metrics=True: Dict with results and quality metrics
            
        Examples:
            # Get formatted context (default)
            >>> context = engine.query("what is ML?")
            
            # Get structured output with quality metrics
            >>> result = engine.query("what is ML?", metrics=True, rerank=True)
            >>> print(result['quality']['confidence'])  # 0.85
        """
        results = self.search(question, top_k=top_k, mode=mode, rerank=rerank)
        
        if not results:
            if metrics:
                return {
                    "query": question,
                    "results": [],
                    "count": 0,
                    "quality": {
                        "confidence": 0.0,
                        "quality": "no_results",
                        "warning": "No results found",
                        "metrics": {}
                    }
                }
            return "No relevant information found."
        
        # If metrics requested, return structured output
        if metrics:
            from radiate.metrics import QualityMetrics
            quality_data = QualityMetrics.analyze_retrieval(results)
            return {
                "query": question,
                "results": results,
                "count": len(results),
                "quality": quality_data
            }
        
        # Default: Format as string (backward compatible)
        context_parts = []
        for r in results:
            # Determine score label and value
            if "rerank_score" in r:
                score_label = "Rerank"
                score_value = r["rerank_score"]
            elif mode == "hybrid" and "rrf_score" in r:
                score_label = "RRF"
                score_value = r["rrf_score"]
            else:
                score_label = "Score"
                score_value = r.get("score", 0)
            
            context_parts.append(
                f"[Source: {r['source']}, Chunk {r['chunk_index']}, "
                f"{score_label}: {score_value:.4f}]\n{r['text']}"
            )
        
        return "\n\n".join(context_parts)


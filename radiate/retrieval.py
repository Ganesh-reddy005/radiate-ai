"""
Hybrid retrieval combining dense vector search with BM25 sparse retrieval.
"""

import math
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict


class BM25:
    """
    BM25 ranking function for text retrieval.
    
    BM25 is a probabilistic ranking function that scores documents based on
    query term frequency, document length, and inverse document frequency.
    
    Attributes:
        k1: Controls term frequency saturation (default: 1.5)
        b: Controls length normalization (default: 0.75)
    
    References:
        Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 recommended)
            b: Length normalization parameter (0-1, where 1 = full normalization)
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.doc_lengths = []
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization by splitting on whitespace and lowercasing.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        return text.lower().split()
    
    def fit(self, corpus: List[str]) -> None:
        """
        Compute IDF scores from corpus.
        
        Args:
            corpus: List of document texts to analyze
        """
        self.corpus_size = len(corpus)
        
        if self.corpus_size == 0:
            return
        
        # Calculate document lengths
        self.doc_lengths = [len(self._tokenize(doc)) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size
        
        # Calculate document frequencies
        df = defaultdict(int)
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1
        
        # Calculate IDF scores using BM25's IDF formula
        self.idf = {}
        for token, freq in df.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[token] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )
    
    def get_scores(self, query: str, corpus: List[str]) -> List[float]:
        """
        Calculate BM25 scores for query against corpus.
        
        Args:
            query: Query string
            corpus: List of document texts
        
        Returns:
            List of BM25 scores for each document
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, doc in enumerate(corpus):
            doc_tokens = self._tokenize(doc)
            doc_token_freqs = Counter(doc_tokens)
            doc_len = self.doc_lengths[idx] if idx < len(self.doc_lengths) else len(doc_tokens)
            
            score = 0.0
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                idf_score = self.idf[token]
                tf = doc_token_freqs.get(token, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1)))
                
                score += idf_score * (numerator / denominator)
            
            scores.append(score)
        
        return scores


class HybridRetriever:
    """
    Combines dense vector search with BM25 sparse retrieval using Reciprocal Rank Fusion.
    
    This retriever implements a hybrid search strategy that:
    1. Performs dense vector similarity search (semantic understanding)
    2. Performs BM25 keyword matching (lexical precision)
    3. Fuses results using Reciprocal Rank Fusion for optimal ranking
    
    Attributes:
        radiate: Radiate instance for vector operations
        rrf_k: RRF constant for fusion (default: 60)
        bm25: BM25 scorer instance
    """
    
    def __init__(self, radiate_instance, rrf_k: int = 60):
        """
        Initialize hybrid retriever.
        
        Args:
            radiate_instance: Radiate instance for vector search and database access
            rrf_k: RRF constant (higher = less emphasis on rank position)
        """
        self.radiate = radiate_instance
        self.rrf_k = rrf_k
        self.bm25 = BM25()
    
    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks (same source + chunk_index).
        Keeps the result with the highest score.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results maintaining relevance order
        """
        if not results:
            return results
        
        # Group by unique identifier (source + chunk_index)
        unique_results = {}
        for r in results:
            # Create unique key
            key = f"{r.get('source', '')}_{r.get('chunk_index', 0)}"
            
            # Keep the one with highest score
            if key not in unique_results:
                unique_results[key] = r
            else:
                # Compare scores (handle different score types)
                current_score = r.get('rrf_score', r.get('score', 0))
                existing_score = unique_results[key].get(
                    'rrf_score',
                    unique_results[key].get('score', 0)
                )
                
                if current_score > existing_score:
                    unique_results[key] = r
        
        # Return deduplicated results, maintaining score order
        deduped = list(unique_results.values())
        
        # Sort by score to maintain relevance order
        deduped.sort(
            key=lambda x: x.get('rrf_score', x.get('score', 0)),
            reverse=True
        )
        
        return deduped
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: RRF_score = Σ 1 / (k + rank_i)
        where k is a constant (default 60) and rank_i is the rank in each list.
        
        Args:
            dense_results: Results from vector search
            sparse_results: Results from BM25 search
        
        Returns:
            Merged and reranked results with RRF scores
        """
        # Create rank maps using unique document identifiers
        def get_doc_key(r):
            return f"{r.get('source', '')}_{r.get('chunk_index', 0)}"
        
        dense_ranks = {get_doc_key(r): idx for idx, r in enumerate(dense_results)}
        sparse_ranks = {get_doc_key(r): idx for idx, r in enumerate(sparse_results)}
        
        # Calculate RRF scores
        rrf_scores = {}
        all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        for doc_key in all_keys:
            score = 0.0
            
            if doc_key in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[doc_key] + 1)
            
            if doc_key in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[doc_key] + 1)
            
            rrf_scores[doc_key] = score
        
        # Build document map (prefer dense results for metadata)
        results_map = {}
        for r in dense_results:
            key = get_doc_key(r)
            if key not in results_map:
                results_map[key] = r
        
        for r in sparse_results:
            key = get_doc_key(r)
            if key not in results_map:
                results_map[key] = r
        
        # Create merged results with RRF scores
        merged = []
        for doc_key, score in sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if doc_key in results_map:
                result = results_map[doc_key].copy()
                result['rrf_score'] = score
                merged.append(result)
        
        return merged
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        initial_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            mode: Retrieval mode - "dense", "sparse", or "hybrid"
            initial_k: Number of candidates for BM25 scoring (only used in sparse/hybrid)
        
        Returns:
            List of search results with scores and metadata
            
        Raises:
            ValueError: If mode is not one of "dense", "sparse", or "hybrid"
        """
        if mode == "dense":
            results = self._dense_search(query, top_k)
            return self._deduplicate_results(results)
        
        elif mode == "sparse":
            results = self._sparse_search(query, top_k, initial_k)
            return self._deduplicate_results(results)
        
        elif mode == "hybrid":
            dense_results = self._dense_search(query, initial_k)
            sparse_results = self._sparse_search(query, initial_k, initial_k)
            
            merged = self._reciprocal_rank_fusion(dense_results, sparse_results)
            deduped = self._deduplicate_results(merged)
            return deduped[:top_k]
        
        else:
            raise ValueError(
                f"Unknown mode: '{mode}'. Must be 'dense', 'sparse', or 'hybrid'"
            )
    
    def _dense_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform dense vector search using semantic embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            List of results with vector similarity scores
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
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "source": hit.payload.get("source", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "metadata": {
                    k: v for k, v in hit.payload.items()
                    if k not in ["text", "source", "chunk_index"]
                }
            })
        
        return results
    
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        initial_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 sparse search using keyword matching.
        
        Strategy: Retrieve all documents from database, score with BM25, return top results.
        For large collections, consider implementing a more efficient filtering strategy.
        
        Args:
            query: Search query
            top_k: Number of results to return
            initial_k: Maximum candidates to consider (limits computation)
            
        Returns:
            List of results with BM25 scores
        """
        # Retrieve all documents from collection (or initial_k for efficiency)
        # NOTE: For production with large datasets, implement pagination or filtering
        all_points = self.radiate.qdrant_client.scroll(
            collection_name=self.radiate.collection_name,
            limit=initial_k,
            with_payload=True,
            with_vectors=False  # Don't need vectors for BM25
        )[0]
        
        if not all_points:
            return []
        
        # Build candidates list
        candidates = []
        corpus = []
        
        for point in all_points:
            text = point.payload.get("text", "")
            corpus.append(text)
            candidates.append({
                "id": point.id,
                "text": text,
                "source": point.payload.get("source", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "metadata": {
                    k: v for k, v in point.payload.items()
                    if k not in ["text", "source", "chunk_index"]
                }
            })
        
        # Fit BM25 on corpus and score
        self.bm25.fit(corpus)
        bm25_scores = self.bm25.get_scores(query, corpus)
        
        # Attach scores to candidates
        for idx, candidate in enumerate(candidates):
            candidate['score'] = bm25_scores[idx]
        
        # Sort by BM25 score (descending)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]

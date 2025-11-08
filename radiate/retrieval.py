import math
from typing import List, Dict, Any
from collections import Counter, defaultdict


class BM25:
    """
    BM25 ranking function for text retrieval.
    
    Parameters:
        k1: Controls term frequency saturation (default: 1.5)
        b: Controls length normalization (default: 0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.doc_lengths = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and lowercasing."""
        return text.lower().split()
    
    def fit(self, corpus: List[str]):
        """
        Compute IDF scores from corpus.
        
        Args:
            corpus: List of document texts
        """
        self.corpus_size = len(corpus)
        
        # Calculate document lengths
        self.doc_lengths = [len(self._tokenize(doc)) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculate document frequencies
        df = defaultdict(int)
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1
        
        # Calculate IDF scores
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
    
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
            doc_len = self.doc_lengths[idx]
            
            score = 0.0
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                idf_score = self.idf[token]
                tf = doc_token_freqs.get(token, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                
                score += idf_score * (numerator / denominator)
            
            scores.append(score)
        
        return scores


class HybridRetriever:
    """
    Combines dense vector search with BM25 sparse retrieval using RRF fusion.
    """
    
    def __init__(self, radiate_instance, rrf_k: int = 60):
        """
        Initialize hybrid retriever.
        
        Args:
            radiate_instance: Radiate instance for vector search
            rrf_k: RRF constant (default: 60)
        """
        self.radiate = radiate_instance
        self.rrf_k = rrf_k
        self.bm25 = BM25()
    
    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from vector search
            sparse_results: Results from BM25 search
        
        Returns:
            Merged and reranked results
        """
        # Create rank maps
        dense_ranks = {r['id']: idx for idx, r in enumerate(dense_results)}
        sparse_ranks = {r['id']: idx for idx, r in enumerate(sparse_results)}
        
        # Calculate RRF scores
        rrf_scores = {}
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        for doc_id in all_ids:
            score = 0.0
            
            if doc_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[doc_id] + 1)
            
            if doc_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[doc_id] + 1)
            
            rrf_scores[doc_id] = score
        
        # Merge results with RRF scores
        results_map = {}
        for r in dense_results + sparse_results:
            if r['id'] not in results_map:
                results_map[r['id']] = r
        
        # Sort by RRF score
        merged = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            result = results_map[doc_id].copy()
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
            top_k: Number of final results
            mode: "dense", "sparse", or "hybrid"
            initial_k: Number of candidates to retrieve for BM25 scoring
        
        Returns:
            List of search results
        """
        if mode == "dense":
            # Dense-only (original behavior)
            return self._dense_search(query, top_k)
        
        elif mode == "sparse":
            # Sparse-only (BM25)
            return self._sparse_search(query, top_k, initial_k)
        
        elif mode == "hybrid":
            # Hybrid: RRF fusion
            dense_results = self._dense_search(query, initial_k)
            sparse_results = self._sparse_search(query, initial_k, initial_k)
            
            merged = self._reciprocal_rank_fusion(dense_results, sparse_results)
            return merged[:top_k]
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'dense', 'sparse', or 'hybrid'")
    
    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform dense vector search."""
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
                "metadata": {k: v for k, v in hit.payload.items() 
                           if k not in ["text", "source", "chunk_index"]}
            })
        
        return results
    
    def _sparse_search(
        self, 
        query: str, 
        top_k: int,
        initial_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 sparse search.
        
        Strategy: Retrieve initial_k results via dense search, then rerank with BM25.
        """
        # Get candidate documents via dense search
        candidates = self._dense_search(query, initial_k)
        
        if not candidates:
            return []
        
        # Extract texts and fit BM25
        corpus = [c['text'] for c in candidates]
        self.bm25.fit(corpus)
        
        # Score with BM25
        bm25_scores = self.bm25.get_scores(query, corpus)
        
        # Combine scores with results
        for idx, candidate in enumerate(candidates):
            candidate['score'] = bm25_scores[idx]
        
        # Sort by BM25 score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]

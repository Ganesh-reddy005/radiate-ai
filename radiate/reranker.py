# reranker.py

from typing import List, Optional, Tuple
from sentence_transformers import CrossEncoder
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class OfflineCrossEncoderReranker:
    """
    A flexible reranker using a local HuggingFace/SBERT cross-encoder model.
    
    Args:
        model_name (str): The model to use (default: ms-marco-MiniLM-L-12-v2).
        device (str): Runtime device ('cpu' or 'cuda').
        
    Example usage:
        reranker = OfflineCrossEncoderReranker()
        docs = ["Document 1", "Document 2", "Doc 3"]
        reranked, scores = reranker.rerank("What is AI?", docs, top_k=2, return_scores=True)
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(model_name, device=device)
        
    def rerank(self, query: str, candidate_docs: List[str], top_k: Optional[int] = None, return_scores: bool = False) -> Tuple[List[str], Optional[List[float]]]:
        # Prepare input pairs for scoring
        pairs = [(query, doc) for doc in candidate_docs]
        scores = self.model.predict(pairs)
        
        # Sort candidates by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices if top_k is None else ranked_indices[:top_k]
        
        reranked_docs = [candidate_docs[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        if return_scores:
            return reranked_docs, top_scores
        return reranked_docs

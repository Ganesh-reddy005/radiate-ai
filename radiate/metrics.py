# radiate/metrics.py

"""Quality metrics and scoring for RAG retrieval."""

from typing import List, Dict, Any
import statistics


class QualityMetrics:
    """Calculate quality metrics for retrieval results."""
    
    @staticmethod
    def calculate_confidence(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate confidence score and metrics for retrieval results.
        
        Args:
            results: List of result dicts with scores
            
        Returns:
            Dict with quality metrics
        """
        if not results:
            return {
                "confidence": 0.0,
                "quality": "no_results",
                "warning": "No results found",
                "metrics": {}
            }
        
        # Extract scores (prioritize rerank_score > rrf_score > score)
        scores = []
        for r in results:
            if "rerank_score" in r:
                scores.append(float(r["rerank_score"]))
            elif "rrf_score" in r:
                scores.append(float(r["rrf_score"]))
            elif "score" in r:
                scores.append(float(r["score"]))
        
        if not scores:
            return {
                "confidence": 0.0,
                "quality": "no_scores",
                "warning": "Results have no scores",
                "metrics": {}
            }
        
        # Calculate metrics
        top_score = max(scores)
        avg_score = statistics.mean(scores)
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        score_range = max(scores) - min(scores)
        
        # Calculate confidence (0-1 scale)
        # High confidence = high top score + good score distribution
        confidence = QualityMetrics._calculate_confidence_score(
            top_score, avg_score, score_std, score_range, len(scores)
        )
        
        # Determine quality level
        quality, warning = QualityMetrics._assess_quality(
            confidence, top_score, score_range, len(scores)
        )
        
        return {
            "confidence": round(confidence, 3),
            "quality": quality,
            "warning": warning,
            "metrics": {
                "top_score": round(top_score, 4),
                "avg_score": round(avg_score, 4),
                "score_std": round(score_std, 4),
                "score_range": round(score_range, 4),
                "num_results": len(scores),
                "scores": [round(s, 4) for s in scores]
            }
        }
    
    @staticmethod
    def _calculate_confidence_score(
        top_score: float,
        avg_score: float,
        score_std: float,
        score_range: float,
        num_results: int
    ) -> float:
        """Calculate overall confidence score (0-1)."""
        
        # Normalize top score (assume rerank scores typically range -10 to 10)
        # For dense/sparse scores (0-1), they're already normalized
        if top_score > 1.0:
            # Likely rerank score - normalize to 0-1
            normalized_top = min(max((top_score + 10) / 20, 0), 1)
        else:
            normalized_top = top_score
        
        # Weight factors:
        # - 60% top score (most important - best result quality)
        # - 25% average score (overall quality)
        # - 15% consistency (low std = more consistent results)
        
        consistency_score = 1.0 - min(score_std / (abs(avg_score) + 0.01), 1.0)
        
        confidence = (
            0.60 * normalized_top +
            0.25 * min(max((avg_score + 10) / 20, 0), 1) +
            0.15 * consistency_score
        )
        
        return min(max(confidence, 0), 1)
    
    @staticmethod
    def _assess_quality(
        confidence: float,
        top_score: float,
        score_range: float,
        num_results: int
    ) -> tuple:
        """Assess quality level and generate warning if needed."""
        
        # Quality thresholds
        if confidence >= 0.8:
            quality = "excellent"
            warning = None
        elif confidence >= 0.6:
            quality = "good"
            warning = None
        elif confidence >= 0.4:
            quality = "fair"
            warning = "Consider adding more relevant documents or rephrasing query"
        else:
            quality = "poor"
            warning = "Low confidence - results may not be relevant"
        
        # Additional warnings
        if num_results < 3:
            warning = f"Only {num_results} results found - consider expanding document base"
        elif score_range < 0.1 and top_score < 1.0:
            warning = "All results have similar low scores - may need better documents"
        
        return quality, warning
    
    @staticmethod
    def analyze_retrieval(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive analysis of retrieval quality.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Detailed analysis report
        """
        quality_data = QualityMetrics.calculate_confidence(results)
        
        # Add retrieval analysis
        analysis = {
            **quality_data,
            "retrieval_analysis": {
                "total_chunks": len(results),
                "unique_sources": len(set(r.get("source", "") for r in results)),
                "has_reranking": any("rerank_score" in r for r in results),
                "retrieval_mode": "hybrid" if any("rrf_score" in r for r in results) else "dense"
            }
        }
        
        return analysis




def log_llm_stats(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    latency: float = 0.0,
    cost: float = 0.0
) -> None:
    """
    Log LLM usage statistics.
    
    Args:
        provider: LLM provider name
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency: Response time in seconds
        cost: Estimated cost in USD
    """
    print(f"\n[LLM Stats] {provider}/{model}")
    print(f"  Input tokens: {input_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print(f"  Total tokens: {input_tokens + output_tokens}")
    print(f"  Latency: {latency:.2f}s")
    if cost > 0:
        print(f"  Estimated cost: ${cost:.4f}")

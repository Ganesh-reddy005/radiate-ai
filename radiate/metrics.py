import time
from functools import wraps
from typing import Callable, Any, Dict, List, Optional

class MetricLogger:
    """
    Simple metrics/logging class for ingest, search and LLM.
    Stores metrics in-memory and prints to console.
    """
    def __init__(self):
        self.metrics = []

    def log(self, name: str, value: Any, unit: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        entry = {
            "metric": name,
            "value": value,
            "unit": unit,
            "extra": extra or {},
            "timestamp": time.time()
        }
        self.metrics.append(entry)
        # Print to console — can replace with file/structured logging later
        unit_str = f" {unit}" if unit else ""
        print(f"[METRIC] {name}: {value}{unit_str} {extra or ''}")

    def save_to_file(self, file_path: str):
        import json
        with open(file_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved metrics to {file_path}")

metric_logger = MetricLogger()

def timed(name: str):
    """Decorator to time function execution and log result."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            duration = end - start
            metric_logger.log(f"{name}_duration", round(duration, 4), "sec")
            return result
        return wrapper
    return decorator

def log_ingest_stats(file_path: str, total_chunks: int, avg_chunk_tokens: float, error_count: int = 0):
    metric_logger.log("ingest_file", file_path, extra={
        "total_chunks": total_chunks,
        "avg_chunk_tokens": avg_chunk_tokens,
        "errors": error_count
    })

def log_search_stats(query: str, latency: float, retrieved: int, reranked: int = None):
    data = {
        "retrieved_chunks": retrieved,
        "latency_sec": round(latency, 4)
    }
    if reranked is not None:
        data["reranked_chunks"] = reranked
    metric_logger.log("search_query", query, extra=data)

def log_llm_stats(prompt: str, latency: float, answer_tokens: int):
    metric_logger.log("llm_call", None, extra={
        "prompt_tokens": len(prompt.split()),
        "latency_sec": round(latency, 4),
        "answer_tokens": answer_tokens
    })

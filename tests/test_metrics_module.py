import time
from radiate.core import Radiate
from radiate.metrics import metric_logger, log_ingest_stats, log_search_stats, timed

# --- Decorated ingestion with timing and stats ----

@timed("ingest_file")
def custom_ingest_file(radiate, file_path):
    # Minimal ingest for a single file, returns chunk info
    result = radiate.ingest(file_path)
    # Simulate average chunk size (fake for now; replace with your logic)
    chunks = radiate.get_all_chunks()
    total_tokens = sum(len(c['text'].split()) for c in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0
    log_ingest_stats(file_path, result['total_chunks'], avg_tokens)
    return result

# --- Decorated search with timing and stats ----

@timed("search_query")
def custom_search(radiate, query, top_k=5):
    start = time.time()
    result = radiate.search(query, top_k=top_k)
    end = time.time()
    retrieved = len(result.get('chunks', [])) if isinstance(result, dict) else len(result)
    latency = end - start
    log_search_stats(query, latency, retrieved)
    return result

# --- Main test routine ----

def main():
    print("="*70)
    print("METRICS TEST: INGESTION")
    print("="*70)
    radiate = Radiate(embedding_provider='local')
    radiate.delete_collection(confirm=True)
    # Change file path if needed
    single_file = "test_data/sample.txt"
    custom_ingest_file(radiate, single_file)

    print("\n" + "="*70)
    print("METRICS TEST: SEARCH")
    print("="*70)
    # Simple search query
    query = "What is machine learning?"
    custom_search(radiate, query, top_k=3)

    print("\n" + "="*70)
    print("SAVING ALL METRICS TO FILE")
    print("="*70)
    metric_logger.save_to_file("metrics_test_output.json")

if __name__ == "__main__":
    main()

import time
import asyncio
from radiate import Radiate

def test_sync():
    """Test synchronous ingestion."""
    print("\n" + "="*60)
    print("SYNC INGESTION TEST")
    print("="*60)
    
    radiate = Radiate(embedding_provider="local")
    radiate.delete_collection(confirm=True)
    
    start = time.time()
    result = radiate.ingest("test_data/")
    elapsed = time.time() - start
    
    # Use total_chunks for directory ingestion
    chunks = result.get('total_chunks', result.get('chunks_ingested', 0))
    files = result.get('total_files', 1)
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Files: {files}")
    print(f"  Chunks: {chunks}")
    if chunks > 0:
        print(f"  Speed: {chunks/elapsed:.1f} chunks/sec")
    
    return elapsed

async def test_async():
    """Test asynchronous ingestion."""
    print("\n" + "="*60)
    print("ASYNC INGESTION TEST")
    print("="*60)
    
    radiate = Radiate(embedding_provider="local")
    radiate.delete_collection(confirm=True)
    
    start = time.time()
    result = await radiate.ingest_async("test_data/")
    elapsed = time.time() - start
    
    # Use total_chunks for directory ingestion
    chunks = result.get('total_chunks', result.get('chunks_ingested', 0))
    files = result.get('total_files', 1)
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Files: {files}")
    print(f"  Chunks: {chunks}")
    if chunks > 0:
        print(f"  Speed: {chunks/elapsed:.1f} chunks/sec")
    
    return elapsed

def main():
    print("\n" + "="*60)
    print("ASYNC VS SYNC PERFORMANCE COMPARISON")
    print("="*60)
    
    # Run sync test
    sync_time = test_sync()
    
    # Run async test
    async_time = asyncio.run(test_async())
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Sync time:  {sync_time:.2f}s")
    print(f"Async time: {async_time:.2f}s")
    
    if async_time < sync_time:
        improvement = ((sync_time - async_time) / sync_time) * 100
        speedup = sync_time / async_time
        print(f"\nAsync is {improvement:.1f}% faster ({speedup:.1f}x speedup)")
    else:
        print("\nAsync was not faster (dataset may be too small)")

if __name__ == "__main__":
    main()

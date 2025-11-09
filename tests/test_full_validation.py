"""
Comprehensive validation test for Radiate v2.
Tests all features: embeddings, ingestion, hybrid search, cost tracking.
"""

from radiate import Radiate
import os


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")


def test_initialization():
    """Test 1: Initialize with local embeddings."""
    print_section("TEST 1: Initialization")
    
    try:
        radiate = Radiate(
            embedding_provider="local",
            track_costs=True
        )
        print("✓ Radiate initialized successfully")
        
        # Check collection info
        info = radiate.get_collection_info()
        print(f"✓ Collection: {info['name']}")
        print(f"✓ Vector dimension: {info['vector_dimension']}")
        print(f"✓ Points count: {info['points_count']}")
        
        return radiate
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return None


def test_ingestion(radiate):
    """Test 2: Document ingestion with batch processing."""
    print_section("TEST 2: Document Ingestion")
    
    if not os.path.exists("sample.txt"):
        print("✗ sample.txt not found. Create it first.")
        return False
    
    try:
        result = radiate.ingest("sample.txt")
        
        print(f"✓ File ingested: {result['file']}")
        print(f"✓ Chunks created: {result['chunks_ingested']}")
        print(f"✓ Status: {result['status']}")
        
        # Check embedding stats
        stats = result.get('embedding_stats', {})
        print(f"\nEmbedding Statistics:")
        print(f"  Generated: {stats.get('total_embeddings_generated', 0)}")
        print(f"  Cached: {stats.get('cached_embeddings', 0)}")
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', '0%')}")
        print(f"  Total cost: {stats.get('total_cost', '$0.0000')}")
        
        return True
    except Exception as e:
        print(f"✗ Ingestion failed: {e}")
        return False


def test_dense_search(radiate):
    """Test 3: Dense vector search (semantic)."""
    print_section("TEST 3: Dense Search (Semantic)")
    
    query = "How to handle authentication in APIs?"
    
    try:
        results = radiate.search(query, mode="dense", top_k=3)
        
        print(f"Query: '{query}'")
        print(f"Results: {len(results)}\n")
        
        for idx, result in enumerate(results, 1):
            print(f"{idx}. Score: {result['score']:.4f}")
            print(f"   Source: {result['source']}")
            print(f"   Text: {result['text'][:100]}...")
            print()
        
        return len(results) > 0
    except Exception as e:
        print(f"✗ Dense search failed: {e}")
        return False


def test_sparse_search(radiate):
    """Test 4: Sparse BM25 search (keyword matching)."""
    print_section("TEST 4: Sparse Search (BM25)")
    
    query = "rate limit error 429"
    
    try:
        results = radiate.search(query, mode="sparse", top_k=3)
        
        print(f"Query: '{query}'")
        print(f"Results: {len(results)}\n")
        
        for idx, result in enumerate(results, 1):
            print(f"{idx}. BM25 Score: {result['score']:.4f}")
            print(f"   Source: {result['source']}")
            print(f"   Text: {result['text'][:100]}...")
            print()
        
        return len(results) > 0
    except Exception as e:
        print(f"✗ Sparse search failed: {e}")
        return False


def test_hybrid_search(radiate):
    """Test 5: Hybrid search (BM25 + Dense + RRF)."""
    print_section("TEST 5: Hybrid Search (RRF Fusion)")
    
    query = "API security best practices"
    
    try:
        results = radiate.search(query, mode="hybrid", top_k=3)
        
        print(f"Query: '{query}'")
        print(f"Results: {len(results)}\n")
        
        for idx, result in enumerate(results, 1):
            rrf_score = result.get('rrf_score', 0)
            print(f"{idx}. RRF Score: {rrf_score:.4f}")
            print(f"   Source: {result['source']}")
            print(f"   Text: {result['text'][:100]}...")
            print()
        
        return len(results) > 0
    except Exception as e:
        print(f"✗ Hybrid search failed: {e}")
        return False


def test_query_method(radiate):
    """Test 6: Query method (formatted context)."""
    print_section("TEST 6: Query Method")
    
    question = "What are common API authentication methods?"
    
    try:
        context = radiate.query(question, mode="hybrid", top_k=2)
        
        print(f"Question: '{question}'")
        print(f"\nContext:\n{context[:300]}...")
        
        return len(context) > 0
    except Exception as e:
        print(f"✗ Query method failed: {e}")
        return False


def test_cost_tracking(radiate):
    """Test 7: Cost tracking and caching."""
    print_section("TEST 7: Cost Tracking")
    
    try:
        stats = radiate.get_stats()
        
        print("Cost Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test caching by embedding same text twice
        vec1 = radiate.get_embedding("test caching")
        vec2 = radiate.get_embedding("test caching")  # Should be cached
        
        new_stats = radiate.get_stats()
        print(f"\nAfter duplicate embedding:")
        print(f"  Cache hit rate: {new_stats['cache_hit_rate']}")
        
        return True
    except Exception as e:
        print(f"✗ Cost tracking failed: {e}")
        return False


def test_collection_management(radiate):
    """Test 8: Collection management methods."""
    print_section("TEST 8: Collection Management")
    
    try:
        # List collections
        collections = radiate.list_collections()
        print(f"✓ Available collections: {collections}")
        
        # Get collection info
        info = radiate.get_collection_info()
        print(f"✓ Current collection info:")
        for key, value in info.items():
            print(f"    {key}: {value}")
        
        return True
    except Exception as e:
        print(f"✗ Collection management failed: {e}")
        return False


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*60)
    print("RADIATE V2 COMPREHENSIVE VALIDATION TEST".center(60))
    print("="*60)
    
    results = {}
    
    # Test 1: Initialization
    radiate = test_initialization()
    results['initialization'] = radiate is not None
    
    if not radiate:
        print("\n✗ Cannot continue without successful initialization")
        return
    
    # Test 2: Ingestion
    results['ingestion'] = test_ingestion(radiate)
    
    # Test 3-5: Search modes
    results['dense_search'] = test_dense_search(radiate)
    results['sparse_search'] = test_sparse_search(radiate)
    results['hybrid_search'] = test_hybrid_search(radiate)
    
    # Test 6: Query method
    results['query_method'] = test_query_method(radiate)
    
    # Test 7: Cost tracking
    results['cost_tracking'] = test_cost_tracking(radiate)
    
    # Test 8: Collection management
    results['collection_management'] = test_collection_management(radiate)
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("✓ All systems operational. Ready for production.")
    else:
        print("✗ Some tests failed. Review errors above.")


if __name__ == "__main__":
    run_all_tests()

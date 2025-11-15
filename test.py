"""
Basic smoke tests for Radiate.
Run with: python tests/test_basic.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from radiate.core import Radiate
from radiate.llm import LLMClient


def test_initialization():
    """Test Radiate initialization with different configs."""
    print("\n🧪 Test 1: Initialization")
    
    # Test local embeddings
    radiate_local = Radiate(embedding_provider='local')
    print("✅ Local embeddings initialized")
    
    # Test with reranker
    radiate_rerank = Radiate(embedding_provider='local', enable_reranker=True)
    print("✅ Reranker enabled")
    
    # Test collection info
    info = radiate_local.get_collection_info()
    print(f"✅ Collection info: {info.get('vectors_count', 0)} vectors")


def test_basic_ingestion():
    """Test basic PDF ingestion."""
    print("\n🧪 Test 2: Basic Ingestion")
    
    radiate = Radiate(embedding_provider='local')
    
    # Delete collection to start fresh
    radiate.delete_collection(confirm=True)
    
    # Re-initialize
    radiate = Radiate(embedding_provider='local')
    
    # Ingest
    result = radiate.ingest('test_data/', show_progress=False)
    
    assert result['total_chunks'] > 0, f"No chunks ingested! Result: {result}"
    print(f"✅ Ingested {result['total_chunks']} chunks in {result.get('total_time', 0):.2f}s")


def test_list_operations():
    """Test listing operations."""
    print("\n🧪 Test 3: List Operations")
    
    radiate = Radiate(embedding_provider='local')
    
    # List collections
    collections = radiate.list_collections()
    print(f"✅ Collections: {collections}")
    
    # List sources
    sources = radiate.list_sources()
    print(f"✅ Sources: {len(sources)} source(s)")
    
    # Get all chunks
    chunks = radiate.get_all_chunks(limit=5)
    print(f"✅ Retrieved {len(chunks)} chunks")
    
    if chunks:
        # Get chunk by ID
        chunk_id = chunks[0]['id']
        chunk = radiate.get_chunk_by_id(chunk_id)
        print(f"✅ Retrieved chunk by ID: {chunk_id}")


def test_query_without_metrics():
    """Test basic query (backward compatibility)."""
    print("\n🧪 Test 4: Query Without Metrics")
    
    radiate = Radiate(embedding_provider='local')
    result = radiate.query("what is machine learning", top_k=3, mode='hybrid')
    
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) > 0, "No results returned"
    assert 'text' in result[0], "Results missing 'text' field"
    
    print(f"✅ Query returned {len(result)} results")


def test_query_with_metrics():
    """Test query with quality metrics."""
    print("\n🧪 Test 5: Query With Metrics")
    
    radiate = Radiate(embedding_provider='local')
    result = radiate.query(
        "what is machine learning",
        metrics=True,
        top_k=3,
        mode='hybrid'
    )
    
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert 'quality' in result, "Missing 'quality' field"
    assert 'confidence' in result['quality'], "Missing confidence score"
    
    confidence = result['quality']['confidence']
    quality = result['quality']['quality']
    print(f"✅ Confidence: {confidence:.2f} ({quality})")
    
    if result['quality'].get('warning'):
        print(f"⚠️  Warning: {result['quality']['warning']}")


def test_reranking():
    """Test reranking functionality."""
    print("\n🧪 Test 6: Reranking")
    
    radiate = Radiate(embedding_provider='local', enable_reranker=True)
    
    # Without reranking
    result_no_rerank = radiate.query(
        "what is machine learning",
        metrics=True,
        rerank=False,
        top_k=3
    )
    
    # With reranking
    result_rerank = radiate.query(
        "what is machine learning",
        metrics=True,
        rerank=True,
        top_k=3
    )
    
    conf_no_rerank = result_no_rerank['quality']['confidence']
    conf_rerank = result_rerank['quality']['confidence']
    
    improvement = ((conf_rerank - conf_no_rerank) / max(conf_no_rerank, 0.01) * 100)
    
    print(f"✅ Without reranking: {conf_no_rerank:.2f}")
    print(f"✅ With reranking: {conf_rerank:.2f}")
    print(f"✅ Improvement: {improvement:+.1f}%")


def test_search_modes():
    """Test different search modes."""
    print("\n🧪 Test 7: Search Modes")
    
    radiate = Radiate(embedding_provider='local')
    
    modes = ['dense', 'sparse', 'hybrid']
    for mode in modes:
        result = radiate.search("machine learning", top_k=3, mode=mode)
        print(f"✅ {mode.capitalize()} search: {len(result)} results")


def test_analyze_query():
    """Test analyze_query helper."""
    print("\n🧪 Test 8: Analyze Query")
    
    radiate = Radiate(embedding_provider='local', enable_reranker=True)
    radiate.analyze_query("what is machine learning", rerank=True, top_k=3)
    print("✅ Analyze query completed")


def test_compare_modes():
    """Test compare_modes helper."""
    print("\n🧪 Test 9: Compare Modes")
    
    radiate = Radiate(embedding_provider='local', enable_reranker=True)
    radiate.print_comparison("what is machine learning", top_k=3)
    print("✅ Comparison completed")


def test_stats():
    """Test get_stats."""
    print("\n🧪 Test 10: Get Stats")
    
    radiate = Radiate(embedding_provider='local', track_costs=True)
    stats = radiate.get_stats()
    
    print(f"✅ Stats retrieved:")
    print(f"   - Embeddings generated: {stats.get('total_embeddings_generated', 0)}")
    print(f"   - Estimated cost: ${stats.get('total_embedding_cost', 0):.4f}")


def test_edge_cases():
    """Test edge cases."""
    print("\n🧪 Test 11: Edge Cases")
    
    radiate = Radiate(embedding_provider='local')
    
    # Test with empty query
    try:
        result = radiate.query("", top_k=3)
        print("✅ Empty query handled")
    except Exception as e:
        print(f"⚠️  Empty query failed: {e}")
    
    # Test with very long query
    long_query = "machine learning " * 100
    result = radiate.query(long_query, metrics=True, top_k=3)
    print(f"✅ Long query handled (confidence: {result['quality']['confidence']:.2f})")
    
    # Test with non-existent chunk ID
    try:
        chunk = radiate.get_chunk_by_id(999999999)
        print("⚠️  Non-existent chunk returned something")
    except Exception as e:
        print(f"✅ Non-existent chunk handled: {type(e).__name__}")


def test_llm_integration():
    """Test LLM integration (requires API key)."""
    print("\n🧪 Test 12: LLM Integration")
    
    if not os.environ.get("API_KEY") and not os.environ.get("API_KEY"):
        print("⏭️  Skipping LLM test (no API key)")
        return
    
    try:
        radiate = Radiate(embedding_provider='local', enable_reranker=True)
        llm = LLMClient(provider="openrouter", model="nvidia/nemotron-nano-12b-v2-vl:free")
        
        result = radiate.query("How to start a Bussiness", rerank=True, top_k=3)
        answer = llm.answer("what is machine learning", context_chunks=result, max_tokens=100)
        
        assert 'answer' in answer, "Missing answer field"
        assert answer['answer'] is not None, "Answer is None"
        
        print(f"✅ LLM generated answer ({answer['tokens']['output']} tokens)")
        print(f"   Latency: {answer['latency']:.2f}s")
    except Exception as e:
        print(f"⚠️  LLM test failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("🚀 Running Radiate Test Suite")
    print("=" * 60)
    
    tests = [
       test_initialization,
        test_basic_ingestion,
        test_list_operations,
        test_query_without_metrics,
        test_query_with_metrics,
        test_reranking,
        test_search_modes,
        test_analyze_query,
        test_compare_modes,
        test_stats,
        test_edge_cases,
        test_llm_integration
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

from radiate import Radiate

# Initialize Radiate
radiate = Radiate(embedding_provider='local')

# Test file path
test_file = "test_data/test_md.md"

print("=" * 70)
print("COMPARING CHUNKING MODES")
print("=" * 70)

# Delete collection to start fresh
radiate.delete_collection(confirm=True)

# ============================================================
# TEST 1: TOKEN MODE (default)
# ============================================================
print("\n" + "=" * 70)
print("MODE 1: TOKEN CHUNKING")
print("=" * 70)

result_token = radiate.ingest(test_file, chunk_mode=None)
print(f"\nTotal chunks: {result_token['total_chunks']}")

print("\n--- Token Mode Chunks ---")
chunks_token = radiate.get_all_chunks(limit=10)
for i, chunk in enumerate(chunks_token, 1):
    print(f"\nChunk {i} ({len(chunk['text'].split())} words):")
    print(f"  {chunk['text']}...")  # last 100 chars

# Delete collection for next test
radiate.delete_collection(confirm=True)

# ============================================================
# TEST 2: SMART MODE
# ============================================================
print("\n" + "=" * 70)
print("MODE 2: SMART CHUNKING")
print("=" * 70)

result_smart = radiate.ingest(test_file, chunk_mode="smart")
print(f"\nTotal chunks: {result_smart['total_chunks']}")

print("\n--- Smart Mode Chunks ---")
chunks_smart = radiate.get_all_chunks(limit=10)
for i, chunk in enumerate(chunks_smart, 1):
    print(f"\nChunk {i} ({len(chunk['text'].split())} words):")
    print(f"  {chunk['text']}...")  # last 100 chars

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Token mode chunks: {result_token['total_chunks']}")
print(f"Smart mode chunks: {result_smart['total_chunks']}")
print(f"Difference: {abs(result_token['total_chunks'] - result_smart['total_chunks'])} chunks")

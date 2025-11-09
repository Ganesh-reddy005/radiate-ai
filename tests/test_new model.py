from radiate import Radiate
'''
# Test with local embeddings (free!)
print("=== Testing Radiate with Local Embeddings ===\n")

radiate = Radiate(
    embedding_provider="local",
    track_costs=True
)

# Test ingestion
result = radiate.ingest("sample.txt")

print(f"\nIngestion Results:")
print(f"   Chunks ingested: {result.get('chunks_ingested', 0)}")
print(f"   Status: {result['status']}")

print(f"\n Embedding Stats:")
stats = result['embedding_stats']
for key, value in stats.items():
    print(f"   {key}: {value}")

# Test query
print("\n=== Testing Query ===\n")
answer = radiate.query("test question")
print(f"Answer: {answer[:200]}...")

print(f"\nFinal Stats:")
final_stats = radiate.get_stats()
for key, value in final_stats.items():
    print(f"   {key}: {value}")
'''
from radiate import Radiate

radiate = Radiate(embedding_provider="local")
radiate.reset_collection()  # This deletes old collection

# Now ingest again
result = radiate.ingest("sample.txt")

radiate = Radiate(
    embedding_provider="local",
    collection_name="radiate_docs_local"  # Different name
)

from radiate import Radiate

# Initialize with local embeddings
radiate = Radiate(
    embedding_provider="local",
    collection_name="radiate_docs"
)

# If you get dimension mismatch error, delete and recreate:
# radiate.delete_collection(confirm=True)

# Check collection info
info = radiate.get_collection_info()
print("\nCollection Info:")
for key, value in info.items():
    print(f"  {key}: {value}")

# List all collections
collections = radiate.list_collections()
print(f"\nAvailable collections: {collections}")

# Now test ingestion
result = radiate.ingest("sample.txt")
print(f"\nIngestion: {result['status']}")
print(f"Chunks: {result.get('chunks_ingested', 0)}")

#view all chunks
'''
from radiate.core import Radiate

radiate = Radiate(embedding_provider="local")

# Get first 10 chunks
chunks = radiate.get_all_chunks(limit=10)

print(f"Total chunks retrieved: {len(chunks)}\n")

for chunk in chunks:
    print(f"Chunk {chunk['chunk_index']}: {chunk['text'][:100]}...")
'''

#view chunks from specific format
from radiate import Radiate

# Initialize
radiate = Radiate(embedding_provider="local")

# Delete old collection (without indexes)
radiate.delete_collection(confirm=True)

# Reingest your data (will create collection with indexes)
radiate.ingest("test_data/ml-book.pdf",chunk_mode='smart')

# Now filtering will work
results = radiate.search("authentication", mode="hybrid", top_k=3)
for result in results:
    chunk_id = result['id']
    chunk = radiate.get_chunk_by_id(chunk_id)
    radiate.print_chunk_summary(chunk)

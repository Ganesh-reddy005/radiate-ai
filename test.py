"""from radiate.core import Radiate
radiate=Radiate()
text="I am building a Stripe for RAG"
embedding=radiate.get_embedding(text)
print(embedding)
"""

"""from radiate.ingest import chunk_text
text = "This is a text.." *200
chunks = chunk_text(text , chunk_size=80)
print(f"Split into {len(chunks)} chunks")
print(f"First chunk: {chunks[0][:50]}...")
"""

"""from radiate.ingest import DocumentIngester
from radiate.core import Radiate

radiate = Radiate()
ingester = DocumentIngester(radiate)
result = ingester.ingest_file("sample.txt")
print(result)
"""

from radiate import Radiate

# Initialize
radiate = Radiate()

# Ingest
result = radiate.ingest("sample.txt")
print(result)

# Query
context = radiate.query("What is machine learning?")
print(context)

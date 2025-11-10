from radiate.core import Radiate
import asyncio

radiate = Radiate(embedding_provider='local')

# Test 1: Custom chunk size
print("=" * 70)
print("TEST 1: Custom Chunk Size")
print("=" * 70)
radiate.delete_collection(confirm=True)
result = radiate.ingest(
    "test_data/sample.txt",
    chunk_size=256,
    overlap=25
)
print(f"Chunks: {result['total_chunks']}")

# Test 2: Custom metadata
print("\n" + "=" * 70)
print("TEST 2: Custom Metadata")
print("=" * 70)
radiate.delete_collection(confirm=True)
result = radiate.ingest(
    "test_data/",
    metadata={"project": "radiate", "version": "0.1.0"}
)
print(f"Chunks: {result['total_chunks']}")

# Test 3: Recursive (create nested folders first)
print("\n" + "=" * 70)
print("TEST 3: Recursive Ingestion")
print("=" * 70)
# Create test_data/sub/ with a file
import os
os.makedirs("test_data/sub", exist_ok=True)
with open("test_data/sub/nested.txt", "w") as f:
    f.write("This is a nested file for testing recursive ingestion.")

radiate.delete_collection(confirm=True)
result = radiate.ingest(
    "test_data/",
    recursive=True
)
print(f"Chunks: {result['total_chunks']}")

# Test 4: Skip errors
print("\n" + "=" * 70)
print("TEST 4: Skip Errors")
print("=" * 70)
radiate.delete_collection(confirm=True)
result = radiate.ingest(
    "test_data/",
    skip_errors=True
)
print(f"Success: {result['successful']}, Failed: {result['failed']}")

# Test 5: Async with all params
async def test_async():
    print("\n" + "=" * 70)
    print("TEST 5: Async with All Params")
    print("=" * 70)
    radiate.delete_collection(confirm=True)
    result = await radiate.ingest_async(
        "test_data/",
        chunk_size=512,
        overlap=50,
        metadata={"test": "async"},
        recursive=True,
        skip_errors=True
    )
    print(f"Chunks: {result['total_chunks']}")

asyncio.run(test_async())

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE!")
print("=" * 70)

"""
Basic usage example for Radiate.ai

This example demonstrates:
1. Initializing Radiate
2. Ingesting a document
3. Querying the ingested content
"""

from radiate import Radiate

def main():
    # Initialize with credentials from .env
    radiate = Radiate(validate_connections=True)
    
    # Ingest a single file
    print("Ingesting document...")
    result = radiate.ingest("sample.txt")
    print(f"Ingested {result['chunks_ingested']} chunks from {result['file']}")
    
    # Query the ingested content
    print("\nQuerying: 'What is machine learning?'")
    context = radiate.query("What is machine learning?", top_k=3)
    print("\nRelevant context:")
    print(context)
    
    # Search with scores
    print("\n\nSearching with scores...")
    results = radiate.search("deep learning", top_k=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {result['score']:.3f}")
        print(f"Text: {result['text'][:100]}...")

if __name__ == "__main__":
    main()

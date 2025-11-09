from radiate import Radiate

# Initialize
radiate = Radiate(embedding_provider="local")

# Ingest sample documents
radiate.ingest("sample.txt")

# Test different modes
print("=== Dense Search ===")
results_dense = radiate.search("what is api", mode="dense", top_k=3)
for r in results_dense:
    print(f"Score: {r['score']:.4f} - {r['text'][:100]}")

print("\n=== Sparse (BM25) Search ===")
results_sparse = radiate.search("what is api", mode="sparse", top_k=3)
for r in results_sparse:
    print(f"Score: {r['score']:.4f} - {r['text'][:100]}")

print("\n=== Hybrid Search ===")
results_hybrid = radiate.search("what is api", mode="hybrid", top_k=3)
for r in results_hybrid:
    print(f"RRF Score: {r['rrf_score']:.4f} - {r['text'][:100]}")

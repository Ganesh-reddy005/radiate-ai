# Radiate AI

**A powerful, flexible Python library for building production-ready RAG (Retrieval-Augmented Generation) applications.**

Radiate AI simplifies the process of building intelligent document search and question-answering systems by combining state-of-the-art embeddings, vector databases, and LLM integration into a unified, developer-friendly API[web:1][web:3].

## Key Features

- **Flexible Embedding Providers**: Support for local embeddings (free) and cloud providers (OpenAI)
- **Vector Database Integration**: Built-in Qdrant support for efficient similarity search
- **Hybrid Search**: Dense, sparse, and hybrid retrieval modes for optimal accuracy
- **Smart Reranking**: Cross-encoder reranking for improved result quality
- **Async Operations**: 10x faster ingestion for large datasets with async support
- **LLM Integration**: Built-in client for OpenAI, OpenRouter, and compatible providers
- **Cost Tracking**: Automatic tracking of embedding and LLM costs
- **Developer-Friendly**: Intuitive API with comprehensive error handling and quality metrics

## Installation

```bash
pip install radiate-ai
```

### Required Dependencies

```bash
# Core dependencies (always required)
pip install openai qdrant-client python-dotenv tqdm

# Optional dependencies for local embeddings
pip install sentence-transformers torch
```

## Quick Start

### 1. Setup

Create a `.env` file in your project root:

```env
# Required: Qdrant vector database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Required for embeddings (if using OpenAI)
OPENAI_API_KEY=sk-your-openai-key

# Optional: For LLM answer generation
LLM_API_KEY=your-llm-api-key
```

### 2. Initialize Radiate

```python
from radiate import Radiate

# Initialize with OpenAI embeddings
radiate = Radiate(
    embedding_provider="openai",
    openai_api_key="sk-...",
    collection_name="my_docs"
)
```

### 3. Ingest Documents

```python
# Ingest a directory of documents
result = radiate.ingest(
    "docs/",
    chunk_size=512,
    overlap=50,
    recursive=True
)

print(f"Ingested {result['total_chunks']} chunks from {result['total_files']} files")
```

### 4. Query Your Documents

```python
# Simple query
chunks = radiate.query("What is machine learning?", top_k=3, rerank=True)

# With quality metrics
result = radiate.query(
    "What is machine learning?",
    top_k=3,
    rerank=True,
    metrics=True
)

print(f"Confidence: {result['quality']['confidence']:.2f}")
print(f"Answer preview: {result['results'][0]['text'][:200]}...")
```

### 5. Generate Answers with LLM

```python
from radiate.llm import LLMClient

# Initialize LLM client
llm = LLMClient(
    provider="openai",
    api_key="sk-...",
    model="gpt-3.5-turbo"
)

# Get answer from retrieved context
chunks = radiate.query("What is machine learning?", rerank=True)
response = llm.answer("What is machine learning?", chunks)

print(response['answer'])
print(f"Tokens used: {response['tokens']['total']}")
```

## 📚 Core Concepts

### Embedding Providers

Radiate supports multiple embedding providers for flexibility:

```python
# OpenAI embeddings (recommended for production)
radiate = Radiate(
    embedding_provider="openai",
    openai_api_key="sk-...",
    embedding_model="text-embedding-3-small"  # Optional
)
```

### Retrieval Modes

Choose the retrieval strategy that best fits your use case:

- **`dense`**: Vector similarity search (fast, semantic understanding)
- **`sparse`**: Keyword-based search (exact matches, terminology)
- **`hybrid`**: Combines both approaches (best accuracy)

```python
# Hybrid search with reranking (recommended)
results = radiate.query(
    "explain neural networks",
    mode="hybrid",
    rerank=True,
    top_k=5
)
```

### Reranking

Cross-encoder reranking significantly improves result quality by re-scoring retrieved chunks:

```python
# Enable reranking at initialization
radiate = Radiate(
    embedding_provider="openai",
    enable_reranker=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# Query with reranking
chunks = radiate.query("What is deep learning?", rerank=True)
```

## 🔧 Usage Examples

### Document Ingestion

#### Single File

```python
result = radiate.ingest(
    "documentation.pdf",
    chunk_size=1024,
    overlap=100,
    metadata={"version": "1.0", "type": "technical"}
)
```

#### Directory with Filtering

```python
result = radiate.ingest(
    "docs/",
    pattern="*.md",  # Only markdown files
    chunk_size=512,
    overlap=50,
    recursive=True,
    skip_errors=True
)
```

#### Async Ingestion (10x Faster)

```python
import asyncio

async def ingest_large_dataset():
    result = await radiate.ingest_async(
        "large_docs/",
        chunk_size=1024,
        max_concurrent_files=5,
        show_progress=True
    )
    return result

result = asyncio.run(ingest_large_dataset())
print(f"Ingested {result['total_chunks']} chunks in {result.get('time_taken', 0):.2f}s")
```

### Querying and Search

#### Basic Query

```python
# Get relevant chunks for LLM
chunks = radiate.query("What is transfer learning?", top_k=3)

# Use with LLM
llm = LLMClient(provider="openai")
answer = llm.answer("What is transfer learning?", chunks)
```

#### Query with Metrics

```python
result = radiate.query(
    "Explain backpropagation",
    top_k=5,
    mode="hybrid",
    rerank=True,
    metrics=True
)

# Check quality
if result['quality']['confidence'] > 0.7:
    print("High confidence results")
    for chunk in result['results']:
        print(f"Score: {chunk['score']:.3f}")
        print(f"Text: {chunk['text'][:100]}...")
else:
    print(f"Warning: {result['quality']['warning']}")
```

#### Compare Retrieval Strategies

```python
# Visual comparison of with/without reranking
radiate.print_comparison("What is neural network architecture?")

# Programmatic comparison
comparison = radiate.compare_modes("What is neural network architecture?")
improvement = (
    comparison['with_rerank']['quality']['confidence'] -
    comparison['without_rerank']['quality']['confidence']
)
print(f"Reranking improved confidence by {improvement:.2%}")
```

### LLM Integration

#### Custom System Prompt

```python
llm = LLMClient(provider="openai", model="gpt-4")

chunks = radiate.query("Best practices for model training")

response = llm.answer(
    "Best practices for model training",
    chunks,
    system_prompt="You are an expert ML engineer. Provide concise, actionable advice.",
    temperature=0.3,
    max_tokens=300
)
```

#### Multiple Providers

```python
# OpenAI
llm_openai = LLMClient(provider="openai", model="gpt-3.5-turbo")

# OpenRouter (access to multiple models)
llm_openrouter = LLMClient(
    provider="openrouter",
    api_key="sk-or-...",
    model="anthropic/claude-3-sonnet"
)
```

## 🛠️ Advanced Features

### Collection Management

```python
# List all collections
collections = radiate.list_collections()
print(f"Available collections: {collections}")

# Get collection info
info = radiate.get_collection_info()
print(f"Collection: {info['name']}")
print(f"Vectors: {info['points_count']}")
print(f"Dimension: {info['vector_dimension']}")

# Delete and recreate collection
radiate.delete_collection(confirm=True)
```

### Chunk Inspection

```python
# Get all chunks (paginated)
chunks = radiate.get_all_chunks(limit=10, offset=0)

# Get chunks from specific source
source_chunks = radiate.get_chunks_by_source("api_docs.txt")

# Get specific chunk by ID
chunk = radiate.get_chunk_by_id(123456789)
radiate.print_chunk_summary(chunk)

# List all ingested sources
sources = radiate.list_sources()
print(f"Ingested files: {sources}")
```

### Quality Analysis

```python
# Analyze query quality
radiate.analyze_query(
    "What is convolutional neural network?",
    top_k=5,
    rerank=True
)

# Output:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Confidence: 0.87 (Excellent)
# Top Score: 2.486
# Results: 5 chunks from 2 source(s)
# Reranking: Enabled
# ✓ High quality results
```

### Cost Tracking

```python
# Get embedding statistics
stats = radiate.get_stats()
print(f"Total tokens: {stats.get('total_tokens', 0)}")
print(f"API calls: {stats.get('total_calls', 0)}")
print(f"Estimated cost: ${stats.get('estimated_cost', 0):.4f}")
```

## 🎯 API Reference

### Radiate Class

#### Initialization

```python
Radiate(
    embedding_provider: str = "openai",
    embedding_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "radiate_docs",
    track_costs: bool = True,
    enable_reranker: bool = False,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
)
```

#### Core Methods

| Method                                          | Description                          | Returns           |
|-------------------------------------------------|--------------------------------------|-------------------|
| `ingest(path, **kwargs)`                        | Ingest documents from file/directory | Dict with stats   |
| `ingest_async(path, **kwargs)`                  | Async document ingestion (faster)    | Dict with stats   |
| `query(question, top_k, mode, rerank, metrics)` | Query documents                      | List[Dict] or Dict| 
| `search(query, top_k, mode)`                    | Raw search (for testing)             | List[Dict]        |
| `get_embedding(text)`                           | Generate single embedding            | List[float]       |
| `get_embeddings_batch(texts)`                   | Batch embedding generation           | List[List[float]] |

#### Analysis Methods

| Method                              | Description                    |
|-------------------------------------|--------------------------------|
| `analyze_query(question, **kwargs)` | Print quality analysis         |
| `compare_modes(question, top_k)`    | Compare with/without reranking |
| `print_comparison(question, top_k)` | Print side-by-side comparison  |

#### Collection Methods

| Method                       | Description                    | Returns   |
|------------------------------|--------------------------------|-----------|
| `list_collections()`         | List all collections           | List[str] |
| `get_collection_info()`      | Get collection metadata        | Dict      |
| `delete_collection(confirm)` | Delete and recreate collection | None      |

#### Chunk Methods

| Method                                | Description           | Returns    |
|---------------------------------------|-----------------------|------------|
| `get_all_chunks(limit, offset)`       | Get chunks (paginated)| List[Dict] |
| `get_chunks_by_source(source, limit)` | Get chunks from file  | List[Dict] |
| `get_chunk_by_id(chunk_id)`           | Get specific chunk    | Dict       |
| `list_sources()`                      | List all source files | List[str]  |
| `print_chunk_summary(chunk)`          | Pretty print chunk    | None       |

### LLMClient Class

#### Initialization

```python
LLMClient(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo"
)
```

#### Methods

| Method                                    | Description                   | Returns |
|-------------------------------------------|------------------------------|----------------------------------|
| `answer(query, context_chunks, **kwargs)` | Generate answer from context | Dict with answer, tokens, latency |
| `format_prompt(query, context_chunks)`    | Format prompt for LLM        | str |

## ⚙️ Configuration

### Environment Variables

```env
# Required
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# OpenAI embeddings (if using OpenAI provider)
OPENAI_API_KEY=sk-your-key

# LLM generation (optional)
LLM_API_KEY=your-llm-key
```

### Chunking Strategies

```python
# Smart chunking (sentence-aware, default)
result = radiate.ingest("docs/", chunk_mode="smart", chunk_size=512)

# Token-based chunking (simple split)
result = radiate.ingest("docs/", chunk_mode="token", chunk_size=512)
```

### Performance Tuning

```python
# For large datasets
result = await radiate.ingest_async(
    "large_dataset/",
    chunk_size=1024,           # Larger chunks = fewer API calls
    overlap=100,               # More overlap = better context
    batch_size=64,             # Larger batches = faster processing
    max_concurrent_files=10    # More concurrency = faster ingestion
)

# For accuracy
chunks = radiate.query(
    "complex question",
    top_k=10,                  # More results to rerank
    mode="hybrid",             # Best of both worlds
    rerank=True                # Cross-encoder scoring
)
```

## 📊 Quality Metrics

Radiate provides quality metrics to help you understand retrieval performance:

```python
result = radiate.query("What is AI?", metrics=True)

# Quality assessment
confidence = result['quality']['confidence']  # 0.0 - 1.0
quality_label = result['quality']['quality']  # 'excellent', 'good', 'fair', 'poor'
warning = result['quality']['warning']        # Warning message if low quality

# Retrieval metrics
top_score = result['quality']['metrics']['top_score']
avg_score = result['quality']['metrics']['avg_score']
score_variance = result['quality']['metrics']['score_variance']
```

**Confidence Thresholds:**
- **0.8+**: Excellent (high confidence)
- **0.6-0.8**: Good (reliable)
- **0.4-0.6**: Fair (acceptable)
- **<0.4**: Poor (may not be relevant)

## 🚦 Best Practices

### 1. Choose the Right Embedding Provider

```python
# Production: Use OpenAI for best accuracy
radiate = Radiate(embedding_provider="openai")

# Development/Testing: Use local embeddings (free)
# (Note: Requires sentence-transformers and torch)
```

### 2. Optimize Chunk Size

```python
# Technical docs: Larger chunks for context
radiate.ingest("api_docs/", chunk_size=1024, overlap=100)

# FAQ/Short content: Smaller chunks for precision
radiate.ingest("faq/", chunk_size=256, overlap=50)
```

### 3. Use Hybrid Search + Reranking

```python
# Best accuracy for most use cases
radiate = Radiate(enable_reranker=True)
chunks = radiate.query(question, mode="hybrid", rerank=True)
```

### 4. Monitor Quality

```python
result = radiate.query(question, metrics=True)

if result['quality']['confidence'] < 0.5:
    # Low confidence - adjust strategy
    print(f"Warning: {result['quality']['warning']}")
    # Try: increase top_k, use hybrid mode, or add more documents
```

### 5. Handle Errors Gracefully

```python
# During ingestion
result = radiate.ingest(
    "docs/",
    skip_errors=True,  # Continue on file errors
    show_progress=True # Track progress
)

# During querying
try:
    chunks = radiate.query(question)
    response = llm.answer(question, chunks)
except Exception as e:
    print(f"Error: {e}")
    # Fallback logic
```

## 🔍 Troubleshooting

### Dimension Mismatch Error

If you see: `Dimension mismatch in collection 'radiate_docs'`

**Solution:**
```python
# Option 1: Delete and recreate collection
radiate.delete_collection(confirm=True)

# Option 2: Use a different collection name
radiate = Radiate(collection_name="radiate_docs_v2")
```

### Low Confidence Results

**Possible causes:**
- Insufficient documents in collection
- Query doesn't match document content
- Wrong retrieval mode

**Solutions:**
```python
# 1. Increase top_k
chunks = radiate.query(question, top_k=10)

# 2. Try hybrid mode
chunks = radiate.query(question, mode="hybrid")

# 3. Enable reranking
chunks = radiate.query(question, rerank=True)

# 4. Check what's in your collection
sources = radiate.list_sources()
print(f"Ingested sources: {sources}")
```

### Slow Ingestion

**Solutions:**
```python
# Use async ingestion
result = await radiate.ingest_async(
    "docs/",
    max_concurrent_files=10,
    batch_size=64
)

# Or increase batch size for sync
result = radiate.ingest("docs/", batch_size=64)
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: Describe your use case and proposed solution
3. **Submit PRs**: Fork, create a feature branch, and submit a PR

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/radiate-ai.git
cd radiate-ai

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Qdrant](https://qdrant.tech/) for vector search
- Powered by [OpenAI](https://openai.com/) embeddings
- Inspired by the RAG research community

## 📞 Support

- **Issues**: [https://github.com/ganesh-reddy005/radiate-ai/issues](https://github.com/yourusername/radiate-ai/issues)
- **Discussions**: [https://github.com/ganesh-reddy005/radiate-ai/discussions](https://github.com/yourusername/radiate-ai/discussions)

---

**Made for the AI community**

*Radiate AI - Build intelligent RAG applications in minutes, not days.*

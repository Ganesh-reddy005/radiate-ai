# Radiate Codebase Overview

## Architecture

### Core Components

1. **radiate/core/** - Main orchestration
   - `client.py`: Radiate class (main API)
   - `collection.py`: Collection management methods
   - `stats.py`: Cost tracking and statistics

2. **radiate/embeddings/** - Embedding providers
   - `base.py`: EmbeddingProvider abstract class
   - `local.py`: LocalEmbeddings implementation
   - `openai.py`: OpenAIEmbeddings implementation
   - `factory.py`: create_embeddings() factory

3. **radiate/ingestion/** - Document processing
   - `chunking.py`: chunk_text() function
   - `readers.py`: read_file() function
   - `ingester.py`: DocumentIngester class (sync)
   - `ingester_async.py`: AsyncDocumentIngester class

4. **radiate/retrieval/** - Search functionality
   - `dense.py`: Vector search
   - `sparse.py`: BM25 implementation
   - `hybrid.py`: RRF fusion
   - `query.py`: QueryEngine class

### Data Flow

Ingestion:
File → Reader → Chunker → Embedder → Qdrant

Search:
Query → Embedder → Retriever → RRF → Results

## Making Changes

### Adding New Features

1. Identify component (embedding/ingestion/retrieval)
2. Create new file in appropriate directory
3. Follow existing patterns
4. Update __init__.py exports
5. Add tests
6. Update documentation

### Fixing Bugs

1. Locate relevant module using CODEBASE.md
2. Add test that reproduces bug
3. Fix bug
4. Verify test passes
5. Update CHANGELOG.md

### Current Issues

- None

### Future Improvements

- Reranking (retrieval/reranker.py)
- Async search (retrieval/query_async.py)


## Intelligent Chunking

### Smart Chunking (`smart_chunk_text`)
**Location:** `radiate/ingest.py`

Intelligently chunks text based on file type while respecting logical boundaries:

**Features:**
- **Text files:** Splits on paragraph boundaries (double newline)
- **Markdown files:** Preserves code blocks, headers, and list structures
- **PDF files:** Respects page boundaries when available
- **Fallback:** Uses token-based chunking for oversized paragraphs

**Parameters:**
- `text`: Text to chunk
- `filetype`: File extension ('txt', 'md', 'pdf')
- `chunk_size`: Maximum tokens per chunk (default: 512)
- `overlap`: Token overlap between chunks (default: 50)

**Usage:**
chunks = smart_chunk_text(text, 'md', chunk_size=512, overlap=50)


**Advantages over token chunking:**
- Preserves semantic meaning
- Keeps code blocks intact
- Headers stay with their content
- Complete paragraphs maintained

---

## Ingestion Modes

### Chunk Modes
Radiate supports two chunking strategies:

1. **Smart Mode (default):** Respects logical boundaries
2. **Token Mode:** Fixed token-based splitting

**Smart chunking (default)**
result = radiate.ingest("docs/", chunk_mode="smart")

**Token-based chunking**
result = radiate.ingest("docs/", chunk_mode="token")

---
### Auto-Detection
When no pattern is specified, Radiate auto-detects all supported file types:
**Auto-detects .txt, .md, .pdf**
result = radiate.ingest("docs/")

**Specific pattern**
result = radiate.ingest("docs/", pattern="*.pdf")


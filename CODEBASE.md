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
   - `chunking.py`: `chunk_text()` function
   - `smart_chunk_text()`: intelligent chunking (boundary-aware)
   - `readers.py`: `read_file()` function
   - `ingester.py`: `DocumentIngester` class (sync, all ingest params)
   - `ingester_async.py`: `AsyncDocumentIngester` class (async, all ingest params)

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

## Ingestion: Parameters and Modes

### Flexible Ingestion Parameters  
_All ingestion modes support the following parameters:_

- `chunk_mode`: `"smart"` | `"token"` (default: `"smart"`)  
- `chunk_size`: Max tokens per chunk (default: 512)  
- `overlap`: Tokens overlapping between chunks (default: 50)  
- `metadata`: Custom metadata to attach to all chunks  
- `batch_size`: Embedding batch size for performance tuning (default: 32)  
- `show_progress`: Show progress bar (default: True; requires `tqdm`)  
- `skip_errors`: Continue on file errors (default: False)  
- `recursive`: Scan subdirectories (default: False)  

### Example Usage
```python result = radiate.ingest(
"docs/",
chunk_mode="smart",
chunk_size=256,
overlap=25,
metadata={"project": "radiate", "version": "v0.2"},
batch_size=64,
show_progress=True,
skip_errors=True,
recursive=True
)

### Intelligent Chunking

**Function:** `smart_chunk_text(text, filetype, chunk_size, overlap)`

- Text: splits by paragraph boundaries
- Markdown: preserves code blocks, headers, lists
- PDF: respects page boundaries first, then paragraphs
- If a block exceeds `chunk_size`, fallback to token splitting

### Token Chunking

**Function:** `chunk_text(text, chunk_size, overlap)`
- Splits strictly by tokens, may break mid-paragraph or mid-code block.

---

## Indexing

- Each chunk is indexed with `source` (file path) keyword, and `chunk_index` integer for easy filtering.

---

## Error Handling

- If `skip_errors` is enabled, ingestion continues after errors.
- Progress bars use `tqdm` if installed and `show_progress=True`.

## Future Improvements

- Reranking (retrieval/reranker.py)
- Async search (retrieval/query_async.py)
- More robust ingestion for edge cases

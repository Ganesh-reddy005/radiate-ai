# Quick Module Reference

## Need to modify...

### Embeddings?
→ `embeddings.py` (323 lines)
- `EmbeddingProvider` (base class)
- `LocalEmbeddings` (local models)
- `OpenAIEmbeddings` (OpenAI API)
- `create_embeddings()` (factory)

### Ingestion?
→ `ingest.py` (215 lines) - sync version
→ `ingest_async.py` (169 lines) - async version
- `chunk_text()` - text splitting
- `read_file()` - file reading (.txt, .md, .pdf)
- `DocumentIngester` / `AsyncDocumentIngester`

### Search/Retrieval?
→ `query.py` (110 lines) - main search interface
→ `retrieval.py` (250 lines) - search algorithms
- `QueryEngine.search()` - all search modes
- `BM25` - sparse retrieval
- `HybridRetriever` - RRF fusion

### Core/API?
→ `core.py` (574 lines) - main class
- `Radiate.__init__()` - initialization
- `Radiate.ingest()` / `ingest_async()` - ingestion
- `Radiate.search()` / `query()` - search
- Collection management methods
- Chunk inspection methods

## File Growth Watch List

- `core.py`: 574/600 lines (96% threshold)
  - Action if exceeds 600: Split into core/client.py and core/collection.py



## radiate/ingest.py

### Functions
- `read_file(file_path)` - Read text from .txt, .md, .pdf files
- `chunk_text(text, chunk_size=512, overlap=50)` - Token-based chunking
- `smart_chunk_text(text, filetype, chunk_size=512, overlap=50)` - **NEW** Intelligent chunking

### Classes

#### DocumentIngester
Handles synchronous document ingestion.

**Methods:**
- `ingest_file(file_path, metadata=None, chunk_mode='smart')` - Ingest single file
- `ingest_directory(directory_path, pattern=None, chunk_mode='smart')` - Ingest directory

**Updated:**
- Added `chunk_mode` parameter (smart/token)
- Auto-detection for `pattern=None`
- Standardized return values with `total_chunks`

---

## radiate/ingest_async.py

### Classes

#### AsyncDocumentIngester
Handles asynchronous document ingestion with parallel processing.

**Methods:**
- `ingest_file_async(file_path, metadata=None, chunk_mode='smart')` - Async single file ingestion
- `ingest_directory_async(directory_path, pattern=None, chunk_mode='smart', max_concurrent_files=3)` - Async directory ingestion

**Updated:**
- Added `chunk_mode` parameter throughout
- Pattern auto-detection for multiple file types
- Improved error handling

---

## radiate/core.py

### Radiate Class

**Updated Methods:**

#### `ingest(path, pattern=None, chunk_mode='smart')`
- Added `chunk_mode` parameter
- Standardized return structure (always includes `total_chunks`)
- Auto-detects file types when `pattern=None`

#### `ingest_async(path, pattern=None, chunk_mode='smart', max_concurrent_files=3)`
- Added `chunk_mode` parameter
- Standardized return structure
- Auto-detects file types when `pattern=None`

#### `_ensure_collection_exists()`
- **NEW** Creates payload indexes for `source` (keyword) and `chunk_index` (integer)
- Enables efficient filtering by source file and chunk position

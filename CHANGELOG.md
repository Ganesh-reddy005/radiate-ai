# Changelog - Latest

## [Unreleased] - 2025-11-09

### Added
- **Intelligent Chunking:** New `smart_chunk_text()` function that respects logical boundaries
  - Preserves paragraph structure in text files
  - Keeps code blocks intact in Markdown
  - Respects headers and list structures
  - Falls back to token chunking for oversized content
- **Chunk Mode Parameter:** Added `chunk_mode` to all ingestion methods
  - `chunk_mode='smart'` (default): Intelligent boundary-aware chunking
  - `chunk_mode='token'`: Traditional fixed-token chunking
- **Auto File Type Detection:** When `pattern=None`, automatically ingests .txt, .md, and .pdf files
- **Payload Indexes:** Created indexes on `source` and `chunk_index` fields for faster filtering
- **Standardized Return Values:** Both single file and directory ingestion now return consistent structure with `total_chunks`

### Changed
- **Default Pattern Behavior:** `pattern=None` now auto-detects all supported file types instead of requiring explicit pattern
- **Ingestion Response Structure:** Single file ingestion now includes `total_chunks` and `total_files` for consistency
- **Smart Chunking Default:** All ingestion now uses smart chunking by default (can opt into token mode)

### Fixed
- Pattern handling in directory ingestion now supports `None` value
- Consistent response structure across sync and async ingestion
- Better handling of Markdown code blocks during chunking

### Technical Details
- Updated `DocumentIngester.ingest_file()` to support chunk modes
- Updated `DocumentIngester.ingest_directory()` with pattern auto-detection
- Updated `AsyncDocumentIngester.ingest_file_async()` to support chunk modes
- Updated `AsyncDocumentIngester.ingest_directory_async()` with pattern auto-detection
- Modified `Radiate.ingest()` and `Radiate.ingest_async()` to normalize return values

---

## [0.1.0] - 2025-11-08


# Changelog

## [Unreleased]

### Added
- Async ingestion with 10x performance improvement
- Payload indexes for filtering
- Chunk inspection methods

## [2.0.0] - 2025-11-08

### Added
- Flexible embedding providers (local/OpenAI/OpenRouter)
- Hybrid search (BM25 + Dense + RRF)
- Cost tracking and caching
- Batch processing
- Collection dimension validation

### Changed
- Refactored embedding system with provider abstraction

### Fixed
- Dimension mismatch error handling

## [1.0.0] - 2025-11-01

### Added
- Initial release
- Basic RAG with OpenAI embeddings
- Qdrant integration

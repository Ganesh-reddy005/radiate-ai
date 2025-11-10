import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client.models import PointStruct
from radiate.ingest import chunk_text, read_file, smart_chunk_text

from radiate.ingest import chunk_text, read_file


class AsyncDocumentIngester:
    """Handles async document ingestion with parallel processing."""
    
    def __init__(self, radiate_instance):
        self.radiate = radiate_instance
    
    async def ingest_file_async(
        self, 
        file_path: str, 
        metadata: Dict[str, Any] = None,
        chunk_mode: str = 'smart',
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Async ingest a single file.
        
        Args:
            file_path: Path to file
            metadata: Additional metadata
            chunk_mode: 'smart' or 'token'
            chunk_size: Max tokens per chunk
            overlap: Token overlap between chunks
            
        Returns:
            Ingestion results
        """
        metadata = metadata or {}
        
        try:
            # Read file
            text = read_file(file_path)
            suffix = Path(file_path).suffix.lstrip(".").lower()
            
            # Chunk with specified mode
            if chunk_mode == 'smart':
                chunks = smart_chunk_text(text, suffix, chunk_size=chunk_size, overlap=overlap)
            else:
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            
            if not chunks:
                return {
                    "file": file_path,
                    "chunks_ingested": 0,
                    "status": "skipped",
                    "reason": "No content"
                }
            
            print(f"Processing {file_path} ({len(chunks)} chunks)...")
            
            # Async batch embedding
            embeddings = await self.radiate.embedder.embed_batch_async(
                chunks,
                batch_size=32,
                max_concurrent=5
            )
            
            # Create points
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_metadata = {
                    "source": str(file_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    **metadata
                }
                
                point = PointStruct(
                    id=uuid.uuid4().int & (2**63 - 1),
                    vector=embedding,
                    payload={"text": chunk, **point_metadata}
                )
                points.append(point)
            
            # Upsert to Qdrant
            self.radiate.qdrant_client.upsert(
                collection_name=self.radiate.collection_name,
                points=points
            )
            
            print(f"Ingested {len(chunks)} chunks from {Path(file_path).name}")
            
            return {
                "file": file_path,
                "chunks_ingested": len(chunks),
                "status": "success"
            }
        
        except Exception as e:
            print(f"Failed to ingest {file_path}: {str(e)}")
            return {
                "file": file_path,
                "chunks_ingested": 0,
                "status": "failed",
                "error": str(e)
            }

   #----------------------------------------------------------------------------------------------------------------- 
    async def ingest_directory_async(
        self, 
        directory_path: str, 
        pattern: str = None,
        chunk_mode: str = 'smart',
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: Dict[str, Any] = None,
        max_concurrent_files: int = 3,
        show_progress: bool = True,
        skip_errors: bool = False,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Async ingest directory with parallel processing.
        
        Args:
            directory_path: Path to directory
            pattern: File pattern (None = auto-detect)
            chunk_mode: 'smart' or 'token'
            chunk_size: Max tokens per chunk
            overlap: Token overlap
            metadata: Custom metadata
            max_concurrent_files: Max concurrent file processing
            show_progress: Show progress
            skip_errors: Continue on errors
            recursive: Scan subdirectories
            
        Returns:
            Ingestion results
        """
        path = Path(directory_path)
        
        if not path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Pattern handling
        if pattern is None or pattern == "*":
            patterns = ["*.txt", "*.md", "*.pdf"]
        else:
            patterns = [pattern] if isinstance(pattern, str) else pattern
        
        # Collect files
        files = []
        for pat in patterns:
            if recursive:
                files.extend(path.rglob(pat))
            else:
                files.extend(path.glob(pat))
        
        files = list(set(files))
        
        if not files:
            raise ValueError(
                f"No files matching {patterns} in {directory_path}"
            )
        
        print(f"\nIngesting {len(files)} files from {directory_path}")
        print(f"File types: {', '.join(patterns)}")
        if recursive:
            print("Mode: Recursive")
        print()

        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "details": []
        }
        
        # Process files concurrently
        semaphore = asyncio.Semaphore(max_concurrent_files)
        
        async def process_file(file_path):
            async with semaphore:
                return await self.ingest_file_async(
                    str(file_path),
                    metadata=metadata,
                    chunk_mode=chunk_mode,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
        
        # Run all file ingestions
        file_results = await asyncio.gather(
            *[process_file(f) for f in files],
            return_exceptions=True if skip_errors else False
        )
        
        # Aggregate results
        for result in file_results:
            if isinstance(result, Exception):
                results["failed"] += 1
                results["details"].append({
                    "error": str(result),
                    "status": "failed"
                })
            elif result["status"] == "success":
                results["successful"] += 1
                results["total_chunks"] += result["chunks_ingested"]
                results["details"].append(result)
            else:
                results["failed"] += 1
                results["details"].append(result)
        
        print(f"\nIngestion complete!")
        print(f"  Files: {results['successful']}/{results['total_files']}")
        print(f"  Total chunks: {results['total_chunks']}")
        
        return results

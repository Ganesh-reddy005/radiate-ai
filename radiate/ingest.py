import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
import tiktoken
from qdrant_client.models import PointStruct


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into chunks using tiktoken tokenizer.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in tokens
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    
    return chunks


def read_file(file_path: str) -> str:
    """
    Read text content from a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is unsupported or file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")
    
    if path.suffix == ".txt":
        with open(path, 'r', encoding="utf-8") as f:
            return f.read()
    
    elif path.suffix == '.md':
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif path.suffix == '.pdf':
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ValueError(
                "PyPDF2 not installed. Install with: pip install PyPDF2"
            )
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: .txt, .md, .pdf"
        )


class DocumentIngester:
    """Handles document ingestion into Qdrant with batch optimization."""
    
    def __init__(self, radiate_instance):
        """
        Initialize ingester with a Radiate instance.
        
        Args:
            radiate_instance: Radiate class instance for API access
        """
        self.radiate = radiate_instance
    
    def ingest_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ingest a single file into Qdrant with batch embedding.
        
        Args:
            file_path: Path to file to ingest
            metadata: Additional metadata to store with chunks
            
        Returns:
            Dictionary with ingestion results
        """
        metadata = metadata or {}
        
        try:
            text = read_file(file_path)
            chunks = chunk_text(text)
            
            if not chunks:
                return {
                    "file": file_path,
                    "chunks_ingested": 0,
                    "status": "skipped",
                    "reason": "No content to ingest"
                }
            
            # Batch embed all chunks (much faster!)
            print(f"Processing {file_path} ({len(chunks)} chunks)...")
            embeddings = self.radiate.get_embeddings_batch(chunks)
            
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_metadata = {
                    "source": str(file_path),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    **metadata
                }
                
                point = PointStruct(
                    id=uuid.uuid4().int & (2**63 - 1),  # Ensure positive int64
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
    
    def ingest_directory(
        self, 
        directory_path: str, 
        pattern: str = "*.txt"
    ) -> Dict[str, Any]:
        """
        Ingest all files matching pattern in a directory.
        
        Args:
            directory_path: Path to directory
            pattern: File pattern (e.g., "*.txt", "*.md", "*.pdf")
            
        Returns:
            Dictionary with overall ingestion results
        """
        path = Path(directory_path)
        
        if not path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        files = list(path.glob(pattern))
        
        if not files:
            raise ValueError(
                f"No files matching '{pattern}' found in {directory_path}"
            )
        
        print(f"\nIngesting {len(files)} files from {directory_path}")
        print(f"Pattern: {pattern}\n")
        
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "details": []
        }
        
        for file_path in files:
            result = self.ingest_file(str(file_path))
            
            if result["status"] == "success":
                results["successful"] += 1
                results["total_chunks"] += result["chunks_ingested"]
            else:
                results["failed"] += 1
            
            results["details"].append(result)
        
        print(f"\nIngestion complete!")
        print(f"   Files processed: {results['successful']}/{results['total_files']}")
        print(f"   Total chunks: {results['total_chunks']}")
        
        return results

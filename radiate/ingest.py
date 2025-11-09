import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
import tiktoken
from qdrant_client.models import PointStruct
import re


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


def smart_chunk_text(text, filetype, chunk_size=512, overlap=50):
        """
        Intelligently chunk text based on filetype.
        - Text/Markdown: Paragraphs/headings/code
        - PDF: Page-aware if the content is provided as one page per string
        Falls back to token chunking for very long chunks.
        """
        def tokenize(t):
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return encoding.encode(t)
        def detokenize(toks):
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            return encoding.decode(toks)
        
        # Split by page if PDF (usually \f is page break but often loader already handles this)
        if filetype.lower() == "pdf":
            blocks = text.split("\f")
        else:
            blocks = [text]
        
        final_chunks = []
        for block in blocks:
            # For Markdown files, keep code blocks together, split on double newline otherwise
            if filetype.lower() == "md":
                # Code block aware paragraph splitting
                paras = []
                in_code = False
                curr = []
                for line in block.splitlines(keepends=True):
                    if re.match(r"^\s*```", line):
                        in_code = not in_code
                        curr.append(line)
                        if not in_code:
                            paras.append("".join(curr))
                            curr = []
                        continue
                    if not in_code and line.strip() == "":
                        if curr:
                            paras.append("".join(curr))
                            curr = []
                    else:
                        curr.append(line)
                if curr: paras.append("".join(curr))
            else:
                # TXT, PDF, or fallback: split on double newline (paragraph boundary)
                paras = re.split(r"\n\s*\n", block)
            
            # Group chunks up to chunk_size
            current = []
            current_toks = 0
            for para in paras:
                n_toks = len(tokenize(para))
                if (current_toks + n_toks < chunk_size) and not para.strip().startswith("#"):
                    current.append(para)
                    current_toks += n_toks
                else:
                    if current:
                        final_chunks.append("\n\n".join(current))
                    current = [para]
                    current_toks = n_toks
            if current:
                final_chunks.append("\n\n".join(current))
            
            # Now ensure chunks aren't too long, break into tokens if needed
        real_final = []
        for chunk in final_chunks:
            toks = tokenize(chunk)
            start = 0
            while start < len(toks):
                real_final.append(detokenize(toks[start:start+chunk_size]))
                if start + chunk_size >= len(toks):
                    break
                start += chunk_size - overlap
        return [c for c in real_final if c.strip()]

class DocumentIngester:
    """Handles document ingestion into Qdrant with batch optimization."""
    
    def __init__(self, radiate_instance):
        """
        Initialize ingester with a Radiate instance.
        
        Args:
            radiate_instance: Radiate class instance for API access
        """
        self.radiate = radiate_instance
    
    def ingest_file(self, file_path: str, metadata: Dict[str, Any] = None, chunk_mode:str ='smart') -> Dict[str, Any]:
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
            suffix = Path(file_path).suffix.lower().strip(".")
            
            if chunk_mode == "smart":
                chunks = smart_chunk_text(text, suffix)
            elif chunk_mode in [None, "Null"]:
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
        pattern: str = None,
        chunk_mode: str = 'smart'
    ) -> Dict[str, Any]:
        """
        Ingest all files matching pattern in a directory.
        
        Args:
            directory_path: Path to directory
            pattern: File pattern (None = auto-detect .txt, .md, .pdf)
            chunk_mode: 'smart' or 'token' chunking mode
            
        Returns:
            Dictionary with overall ingestion results
        """
        path = Path(directory_path)
        
        if not path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # If no pattern specified, auto-detect all supported types
        if pattern is None or pattern == "*":
            patterns = ["*.txt", "*.md", "*.pdf"]
        else:
            patterns = [pattern] if isinstance(pattern, str) else pattern
        
        # Collect files from all patterns
        files = []
        for pat in patterns:
            files.extend(path.glob(pat))
        
        # Remove duplicates
        files = list(set(files))
        
        if not files:
            raise ValueError(
                f"No files matching {patterns} found in {directory_path}"
            )
        
        print(f"\nIngesting {len(files)} files from {directory_path}")
        print(f"File types: {', '.join(patterns)}\n")
        
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "details": []
        }
        
        for file_path in files:
            result = self.ingest_file(str(file_path), chunk_mode=chunk_mode)
            
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




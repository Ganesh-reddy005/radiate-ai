"""
Directory ingestion example for Radiate.ai

Demonstrates batch ingestion of multiple files from a directory.
"""

from radiate import Radiate

def main():
    radiate = Radiate()
    
    # Ingest all markdown files from a directory
    print("Ingesting all .md files from docs/")
    result = radiate.ingest("./docs", pattern="*.md")
    
    print(f"\nIngestion Summary:")
    print(f"Total files: {result['total_files']}")
    print(f"Successful: {result['successful']}")
    print(f"Failed: {result['failed']}")
    
    for detail in result['details']:
        if detail['status'] == 'success':
            print(f"  ✓ {detail['file']}: {detail['chunks_ingested']} chunks")
        else:
            print(f"  ✗ {detail['file']}: {detail['error']}")

if __name__ == "__main__":
    main()

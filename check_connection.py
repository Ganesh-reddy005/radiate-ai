from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient


load_dotenv()

qdrnt_url=os.getenv("QDRANT_URL")
qdrnt_api_key=os.getenv("QDRANT_API_KEY")

print(f"Testing connecton:{qdrnt_url}")

client=QdrantClient(url=qdrnt_url,api_key=qdrnt_api_key)

collections=client.get_collections()

print(f"Connected! Found {len(collections.collections)} Collections")
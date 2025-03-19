import json
import boto3
from chromadb import Client
from pinecone import Pinecone

# AWS S3
s3_client = boto3.client("s3", region_name="us-east-2")
BUCKET_NAME = "aibucket-riya"

# ChromaDB & Pinecone
chroma_client = Client(path="./chroma_db")
pinecone_client = Pinecone(api_key="YOUR_PINECONE_API_KEY")
pinecone_index = pinecone_client.Index("nvidia-embeddings")

def retrieve_documents(query, method="ChromaDB"):
    if method == "ChromaDB":
        results = chroma_client.search(query, top_k=5)
    elif method == "Pinecone":
        results = pinecone_index.query(vector=query, top_k=5)

    return results


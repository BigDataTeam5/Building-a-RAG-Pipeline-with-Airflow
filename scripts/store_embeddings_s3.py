import boto3
import json
from chromadb import Client
from pinecone import Pinecone, ServerlessSpec

# AWS Config
s3_client = boto3.client("s3", region_name="us-east-2")
BUCKET_NAME = "aibucket-riya"

# ChromaDB & Pinecone Config
chroma_client = Client(path="./chroma_db")
pinecone_client = Pinecone(api_key="YOUR_PINECONE_API_KEY")
pinecone_index = pinecone_client.Index("nvidia-embeddings")

def store_embeddings():
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="embeddings/")
    
    for obj in response.get("Contents", []):
        key = obj["Key"]
        embedding_data = json.loads(s3_client.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read().decode("utf-8"))

        # Store in ChromaDB
        chroma_client.insert(embedding_data)

        # Store in Pinecone
        pinecone_index.upsert([(key, embedding_data)])

    print("✅ Embeddings stored in ChromaDB & Pinecone")

store_embeddings()

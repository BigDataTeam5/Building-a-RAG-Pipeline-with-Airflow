import boto3
from sentence_transformers import SentenceTransformer

s3_client = boto3.client("s3", region_name="us-east-2")
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings():
    response = s3_client.list_objects_v2(Bucket="aibucket-riya", Prefix="parsed_reports/")
    for obj in response.get('Contents', []):
        text = s3_client.get_object(Bucket="aibucket-riya", Key=obj['Key'])['Body'].read().decode('utf-8')
        embeddings = model.encode(text)
        s3_client.put_object(Bucket="aibucket-riya", Key=f"embeddings/{obj['Key']}.json", Body=str(embeddings))

generate_embeddings()

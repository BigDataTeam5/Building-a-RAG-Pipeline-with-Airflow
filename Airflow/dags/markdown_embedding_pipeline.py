import os
import tempfile
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from sentence_transformers import SentenceTransformer
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from dotenv import load_dotenv
import re
import shutil
from utils.chunking import KamradtModifiedChunker  # Assuming Kamradt chunking is available
from pinecone import Pinecone, ServerlessSpec
import pinecone
import uuid

# Load configuration
with open('/opt/airflow/config/nvidia_config.json') as config_file:
    config = json.load(config_file)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}

# Combine all tasks into a single DAG
dag = DAG(
    "markdown_pipeline_dag",
    default_args=default_args,
    description="Download, process, and store markdown files with embeddings",
    schedule_interval='@daily',
    catchup=False,
)

def list_and_download_markdown_files(**context):
    load_dotenv('/opt/airflow/.env')
    AWS_CONN_ID = config['AWS_CONN_ID']
    BUCKET_NAME = config['BUCKET_NAME']
    
    s3_bucket_pattern = re.compile(r'^[a-zA-Z0-9.\-_]{1,255}$')
    if not s3_bucket_pattern.match(BUCKET_NAME):
        print(f"WARNING: Invalid bucket name format: '{BUCKET_NAME}'")
        BUCKET_NAME = re.sub(r'[^a-zA-Z0-9.\-_]', '-', BUCKET_NAME)
        print(f"Auto-corrected bucket name to: '{BUCKET_NAME}'")
    
    S3_MISTRAL_OUTPUT = config['S3_MISTRAL_OUTPUT']
    print(f"Using S3 configuration: Bucket={BUCKET_NAME}, Source={S3_MISTRAL_OUTPUT}")
    
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    markdown_files = []
    
    try:
        print(f"Listing keys in bucket '{BUCKET_NAME}' with prefix '{S3_MISTRAL_OUTPUT}'")
        keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=S3_MISTRAL_OUTPUT)
        
        if not keys:
            print(f"No keys found in bucket '{BUCKET_NAME}' with prefix '{S3_MISTRAL_OUTPUT}'")
            return []
            
        md_files = [key for key in keys if key.endswith('.md')]
        print(f"Found {len(md_files)} markdown files in S3")
        
        if not md_files:
            print("No markdown files found in the specified S3 location")
            return []
        
        # Create temp directory for downloads
        temp_dir = tempfile.mkdtemp(prefix="airflow_markdown_")
        print(f"Created temporary directory for downloads: {temp_dir}")
        
        for file_path in md_files:
            # Extract year and quarter information from path
            path_parts = file_path.split('/')
            if len(path_parts) >= 3:
                year = path_parts[-2] if path_parts[-2].isdigit() else "unknown"
                file_name = path_parts[-1]  # Don't lowercase the filename here
                
                # Fix quarter detection by extracting directly from filename
                if file_name == "Q1.md" or file_name == "q1.md":
                    quarter = "Q1"
                elif file_name == "Q2.md" or file_name == "q2.md":
                    quarter = "Q2"
                elif file_name == "Q3.md" or file_name == "q3.md":
                    quarter = "Q3" 
                elif file_name == "Q4.md" or file_name == "q4.md":
                    quarter = "Q4"
                else:
                    quarter = "unknown"
            else:
                year = "unknown"
                quarter = "unknown"
            
            # Create a clean local path that mirrors the S3 structure
            local_file_path = os.path.join(temp_dir, file_path)
            local_dir = os.path.dirname(local_file_path)
            
            # Ensure the directory exists
            os.makedirs(local_dir, exist_ok=True)
            
            try:
                # Download the file directly to the final path
                object_data = s3_hook.get_key(key=file_path, bucket_name=BUCKET_NAME).get()['Body'].read()
                
                # Write the data to the local file
                with open(local_file_path, 'wb') as f:
                    f.write(object_data)
                    
                print(f"Downloaded {file_path} to {local_file_path}")
                print(f"Detected year: {year}, quarter: {quarter} for file {file_name}")
                
                markdown_files.append({
                    "year": year,
                    "quarter": quarter,
                    "s3_file_path": file_path,
                    "local_file_path": local_file_path,
                    "file_name": os.path.basename(file_path)
                })
            except Exception as e:
                print(f"Error downloading file {file_path}: {str(e)}")
        
        print(f"Downloaded {len(markdown_files)} markdown files to {temp_dir}")
        
        # Store the temp directory and file info for downstream tasks
        context['ti'].xcom_push(key='temp_directory', value=temp_dir)
        context['ti'].xcom_push(key='markdown_files', value=markdown_files)
        return markdown_files
    except Exception as e:
        print(f"Error listing or downloading files from S3: {str(e)}")
        return []

def kamradt_chunking(**context):
    ti = context['ti']
    markdown_files = ti.xcom_pull(key='markdown_files', task_ids='list_and_download_markdown_files')
    if not markdown_files:
        print("No markdown files found for chunking.")
        return
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embedding_function(texts):
        return [model.encode(text).tolist() for text in texts]
    
    chunker = KamradtModifiedChunker(avg_chunk_size=300, min_chunk_size=50, embedding_function=embedding_function)
    
    all_chunks_info = []
    
    for entry in markdown_files:
        year = entry['year']
        quarter = entry['quarter']
        file_path = entry['local_file_path']
        file_name = entry['file_name']
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks = chunker.split_text(content)
            print(f"File {file_path} split into {len(chunks)} chunks using Kamradt chunking.")
            
            chunks_info = {
                "s3_file_path": entry['s3_file_path'],
                "local_file_path": file_path,
                "file_name": file_name,
                "year": year,
                "quarter": quarter,
                "chunks": chunks
            }
            all_chunks_info.append(chunks_info)
            
            ti.xcom_push(key=f'chunks_{file_name}', value=chunks)
        except Exception as e:
            print(f"Error chunking {file_path}: {str(e)}")
    
    ti.xcom_push(key='all_chunks_info', value=all_chunks_info)


def store_chunks_to_pinecone(**context):
    load_dotenv('/opt/airflow/.env')
    ti = context['ti']
    all_chunks_info = ti.xcom_pull(key='all_chunks_info', task_ids='kamradt_chunking')
    if not all_chunks_info:
        print("No chunk info available for Pinecone processing.")
        return
    
    # Get S3 hook and config
    AWS_CONN_ID = config['AWS_CONN_ID']
    BUCKET_NAME = config['BUCKET_NAME']
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    # Set up Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")  # Example: "gcp-starter"
    index_name = "nvidia-reports"

    # Init Pinecone client with new API
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Create index if it doesn't exist
    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        # delete the index if it already exists
        pc.delete_index(index_name)
        print(f"Deleted existing index: {index_name}")
    
    # Create a new index
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created index: {index_name}")
    
    index = pc.Index(index_name)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Group chunks by namespace
    namespace_chunks = {}
    
    for file_info in all_chunks_info:
        year = file_info['year']
        quarter = file_info['quarter'].lower()
        namespace = f"{year}{quarter}" if year != "unknown" and quarter != "unknown" else "unknown"
        
        # Initialize namespace dict if not exists
        if namespace not in namespace_chunks:
            namespace_chunks[namespace] = []
        
        # Add this file's chunks to the namespace collection
        namespace_chunks[namespace].append(file_info)
    
    # Process each namespace
    for namespace, files_info in namespace_chunks.items():
        try:
            print(f"Processing namespace: {namespace}")
            
            # Create JSON for all chunks in this namespace
            json_filename = f"{namespace}.json"
            s3_json_path = f"chunks/{json_filename}"

            sample_files = [f['file_name'] for f in files_info[:5]]
            print(f"Processing {len(files_info)} files in namespace '{namespace}' with sample files: {sample_files}")

            
            # Prepare chunks data
            chunks_data = {}
            
            for file_info in files_info:
                file_name = file_info['file_name']
                chunks = file_info['chunks']
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file_name}_{i}"
                    chunks_data[chunk_id] = {
                        "text": chunk,
                        "file_name": file_name,
                        "year": file_info['year'],
                        "quarter": file_info['quarter'],
                        "chunk_index": i
                    }
            
            # Save JSON to temporary file then upload to S3
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_json:
                json.dump(chunks_data, temp_json, ensure_ascii=False, indent=2)
                temp_json_path = temp_json.name
            
            # Upload JSON file to S3
            s3_hook.load_file(
                filename=temp_json_path,
                key=s3_json_path,
                bucket_name=BUCKET_NAME,
                replace=True
            )
            os.unlink(temp_json_path)  # Clean up temp file
            
            print(f"Uploaded chunks JSON to s3://{BUCKET_NAME}/{s3_json_path}")
            
            # Now create vectors for Pinecone with references to S3 instead of full text
            vectors = []
            for file_info in files_info:
                file_name = file_info['file_name']
                chunks = file_info['chunks']
                s3_path = file_info['s3_file_path']
                year = file_info['year']
                quarter = file_info['quarter']
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file_name}_{i}"
                    embedding = model.encode(chunk).tolist()
                    vector_id = str(uuid.uuid4())
                    
                    # Create metadata WITHOUT the full text
                    metadata = {
                        "year": year,
                        "quarter": quarter.upper(),
                        "source": s3_path,
                        "file_name": file_name,
                        "chunk_index": i,
                        "json_source": f"s3://{BUCKET_NAME}/{s3_json_path}",
                        "chunk_id": chunk_id,
                        "text_preview": chunk[:100] if len(chunk) > 100 else chunk  # Just a preview
                    }
                    
                    vectors.append((vector_id, embedding, metadata))
            
            # Batch upload to Pinecone (100 vectors at a time)
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i+100]
                index.upsert(vectors=batch, namespace=namespace)
                print(f"Upserted batch {i//100 + 1}/{(len(vectors)-1)//100 + 1} to namespace '{namespace}'")
        
        except Exception as e:
            print(f"Failed to process namespace {namespace}: {e}")

    return f"Uploaded all chunks to Pinecone index: {index_name}"



def display_first_vectors_from_pinecone(**context):
    load_dotenv('/opt/airflow/.env')
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = "nvidia-reports"

    # Use the new Pinecone API pattern
    pc = Pinecone(api_key=pinecone_api_key)
    
    try:
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            print(f"Index '{index_name}' does not exist.")
            return
            
        index = pc.Index(index_name)
        
        # Get index stats to find namespaces
        index_stats = index.describe_index_stats()
        print(f"Index '{index_name}' has {index_stats.num_vectors} vectors in {index_stats.num_namespaces} namespaces.")
        namespaces = index_stats.namespaces
        
        if not namespaces:
            print("No namespaces found in index.")
            return
        
        for namespace_name in namespaces.keys():
            print(f"\nNamespace: {namespace_name}")
            try:
                # Query with zero vector to get a sample of vectors
                response = index.query(
                    vector=[0.0] * 384,  # Dimension must match your index
                    top_k=10,
                    include_metadata=True,
                    namespace=namespace_name
                )
                
                print(f"Displaying up to 10 vectors from namespace: {namespace_name}")
                for i, match in enumerate(response.matches):
                    print(f"\n--- Vector {i+1} ---")
                    print(f"ID: {match.id}")
                    print(f"Score: {match.score:.4f}")
                    print(f"Metadata: {match.metadata}")
                    preview = match.metadata.get('text_preview', '')[:150]
                    print(f"Text Preview: {preview}...")
            except Exception as e:
                print(f"Error querying namespace '{namespace_name}': {e}")
    except Exception as e:
        print(f"Error accessing Pinecone: {e}")



# Define the tasks in the DAG
list_and_download_task = PythonOperator(
    task_id="list_and_download_markdown_files",
    python_callable=list_and_download_markdown_files,
    provide_context=True,
    dag=dag,
)

kamradt_chunking_task = PythonOperator(
    task_id="kamradt_chunking",
    python_callable=kamradt_chunking,
    provide_context=True,
    dag=dag,
)


store_to_pinecone_task = PythonOperator(
    task_id="store_chunks_to_pinecone",
    python_callable=store_chunks_to_pinecone,
    provide_context=True,
    dag=dag,
)

display_pinecone_vectors_task = PythonOperator(
    task_id="display_first_vectors_from_pinecone",
    python_callable=display_first_vectors_from_pinecone,
    provide_context=True,
    dag=dag,
)

list_and_download_task >> kamradt_chunking_task >> store_to_pinecone_task >> display_pinecone_vectors_task

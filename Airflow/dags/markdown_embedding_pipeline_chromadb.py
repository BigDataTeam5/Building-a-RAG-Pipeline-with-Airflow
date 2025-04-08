import os
import tempfile
import json
from utils.patching import *
import time
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import chromadb
from chromadb.config import Settings
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import re
import shutil
from utils.chunking import KamradtModifiedChunker  

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
    "markdown_pipeline_chromadb",
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
                file_name = path_parts[-1].lower()
                if 'q1' in file_name.lower():
                    quarter = "Q1"
                elif 'q2' in file_name.lower():
                    quarter = "Q2"
                elif 'q3' in file_name.lower():
                    quarter = "Q3"
                elif 'q4' in file_name.lower():
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
    
    # Create a custom embedding function that doesn't use SentenceTransformer
    def dummy_embedding_function(texts):
        # Return placeholder embeddings of correct dimension (384)
        return [[0.0] * 384 for _ in texts]
    
    # Pass an actual function so KamradtModifiedChunker doesn't use SentenceTransformer
    chunker = KamradtModifiedChunker(
        avg_chunk_size=300, 
        min_chunk_size=50,
        embedding_function=dummy_embedding_function  # This prevents SentenceTransformer from loading
    )
    
    # Rest of your function remains the same
    all_chunks_info = []
    
    
    for entry in markdown_files:
        year = entry['year']
        quarter = entry['quarter']
        file_path = entry['local_file_path']
        file_name = entry['file_name']
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Use the KamradtModifiedChunker from utils.chunking
            chunks = chunker.split_text(content)
            print(f"File {file_path} split into {len(chunks)} chunks using KamradtModifiedChunker.")
            
            chunks_info = {
                "s3_file_path": entry['s3_file_path'],
                "local_file_path": file_path,
                "file_name": file_name,
                "year": year,
                "quarter": quarter,
                "chunks": chunks
            }
            all_chunks_info.append(chunks_info)
        except Exception as e:
            print(f"Error chunking file {file_path}: {str(e)}")
    
    ti.xcom_push(key='all_chunks_info', value=all_chunks_info)
# In markdown_embedding_pipeline_chromadb.py
def process_chunks_to_chromadb(**context):
    ti = context['ti']
    all_chunks_info = ti.xcom_pull(key='all_chunks_info', task_ids='kamradt_chunking')
    
    print("Starting ChromaDB processing...")
    print(f"Got {len(all_chunks_info) if all_chunks_info else 0} files to process")
    
    if not all_chunks_info:
        print("No chunks found to process.")
        return "No chunks to process"
    
    # Use ChromaDB's default embedding function
    ef = embedding_functions.DefaultEmbeddingFunction()
    
    CHROMA_DB_PATH = config['CHROMA_DB_PATH']
    print(f"Using ChromaDB path: {CHROMA_DB_PATH}")
    
    # Ensure directory exists with proper permissions
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        os.chmod(CHROMA_DB_PATH, 0o777)
        print(f"Created/updated ChromaDB directory: {CHROMA_DB_PATH}")
    except Exception as e:
        print(f"Error setting up ChromaDB directory: {str(e)}")
        raise
    
    # Initialize client with explicit settings
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print("Successfully initialized ChromaDB client")
    except Exception as e:
        print(f"Error initializing ChromaDB client: {str(e)}")
        raise

    # Create new collection
    try:
        # Delete existing collection if it exists
        try:
            client.delete_collection("nvidia_embeddings")
            print("Deleted existing collection")
        except Exception as e:
            print(f"No existing collection to delete: {str(e)}")
        
        # Create fresh collection
        collection = client.create_collection(
            name="nvidia_embeddings",
            embedding_function=ef,
            metadata={"description": "NVIDIA markdown documents with embeddings"}
        )
        print("Successfully created new collection")
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        raise

    # Process chunks for each file
    total_chunks_processed = 0
    for file_info in all_chunks_info:
        chunks = file_info['chunks']
        print(f"Processing {len(chunks)} chunks for file {file_info['file_name']}")
        
        try:
            # Add chunks in smaller batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                batch_ids = [f"{file_info['file_name']}_{i+j}" for j in range(len(batch_chunks))]
                batch_metadatas = [
                    {
                        "year": file_info['year'],
                        "quarter": file_info['quarter'],
                        "source": file_info['s3_file_path'],
                        "file_name": file_info['file_name'],
                        "chunk_index": i+j
                    }
                    for j in range(len(batch_chunks))
                ]
                
                print(f"Adding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                collection.add(
                    ids=batch_ids,
                    documents=batch_chunks,
                    metadatas=batch_metadatas
                )
                total_chunks_processed += len(batch_chunks)
                
        except Exception as e:
            print(f"Error processing file {file_info['file_name']}: {str(e)}")
            continue

    print(f"Successfully processed {total_chunks_processed} total chunks")
    return f"Processed {total_chunks_processed} chunks"

def display_first_embeddings(**context):
    CHROMA_DB_PATH = config['CHROMA_DB_PATH']
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = client.get_collection("nvidia_embeddings")
        result = collection.get(limit=10)
        
        print("\n===== FIRST 10 EMBEDDINGS IN COLLECTION =====")
        for i in range(len(result['ids'])):
            doc_id = result['ids'][i]
            metadata = result['metadatas'][i] if result['metadatas'] else {}
            document = result['documents'][i] if result['documents'] else "No document text"
            
            print(f"\n----- Entry {i+1} -----")
            print(f"ID: {doc_id}")
            print(f"Metadata: Year={metadata.get('year', 'N/A')}, Quarter={metadata.get('quarter', 'N/A')}")
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Document preview: {document[:150]}...")
            
            if result['embeddings'] and len(result['embeddings']) > i:
                embedding = result['embeddings'][i]
                print(f"Embedding dimensions: {len(embedding)}")
                print(f"Embedding preview: [{', '.join(f'{v:.4f}' for v in embedding[:5])}...]")
        
        print("\n===== END OF EMBEDDINGS PREVIEW =====")
        return {"total_entries": collection.count(), "displayed_entries": len(result['ids'])}
    except Exception as e:
        print(f"Error displaying embeddings: {str(e)}")
        return {"error": str(e)}

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

process_chunks_task = PythonOperator(
    task_id="process_chunks_to_chromadb",
    python_callable=process_chunks_to_chromadb,
    provide_context=True,
    dag=dag,
)

display_embeddings_task = PythonOperator(
    task_id="display_first_embeddings",
    python_callable=display_first_embeddings,
    provide_context=True,
    dag=dag,
)

list_and_download_task >> kamradt_chunking_task >> process_chunks_task >> display_embeddings_task

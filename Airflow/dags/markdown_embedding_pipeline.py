import os
import tempfile
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from dotenv import load_dotenv
import re
import shutil
from utils.chunking import KamradtModifiedChunker  # Assuming Kamradt chunking is available

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
                file_name = path_parts[-1].lower()
                if 'q1'  or 'Q1' in file_name:
                    quarter = "Q1"
                elif 'q2' or 'Q2'  in file_name:
                    quarter = "Q2"
                elif 'q3' or 'Q3' in file_name:
                    quarter = "Q3"
                elif 'q4' or 'Q4' in file_name:
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

def process_chunks_to_chromadb(**context):
    ti = context['ti']
    all_chunks_info = ti.xcom_pull(key='all_chunks_info', task_ids='kamradt_chunking')
    if not all_chunks_info:
        print("No chunks found for processing.")
        return
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    CHROMA_DB_PATH  = config['CHROMA_DB_PATH']
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection("nvidia_embeddings")
        print("Existing collection deleted successfully")
    except Exception as e:
        print(f"No existing collection to delete or error: {str(e)}")
    
    collection = client.get_or_create_collection(
        name="nvidia_embeddings",
        metadata={"description": "NVIDIA markdown documents with embeddings",
                  "last_refreshed": datetime.now().isoformat()}
    )
    
    total_chunks_processed = 0
    
    for file_info in all_chunks_info:
        year = file_info['year']
        quarter = file_info['quarter']
        chunks = file_info['chunks']
        file_name = file_info['file_name']
        
        try:
            embeddings = [model.encode(chunk).tolist() for chunk in chunks]
            ids = [f"{file_name}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "year": year,
                    "quarter": quarter,
                    "source": file_info['s3_file_path'],
                    "file_name": file_name,
                    "chunk_index": i
                }
                for i in range(len(chunks))
            ]
            
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            
            total_chunks_processed += len(chunks)
            print(f"Processed and stored {len(chunks)} chunks for {file_name}")
        except Exception as e:
            print(f"Error processing chunks for {file_name}: {str(e)}")
    
    ti.xcom_push(key='total_chunks_processed', value=total_chunks_processed)
    
    temp_dir = ti.xcom_pull(key='temp_directory', task_ids='kamradt_chunking')
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {str(e)}")
    
    return "Processing complete"

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

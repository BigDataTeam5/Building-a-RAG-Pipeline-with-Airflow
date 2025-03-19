import os
import tempfile
import shutil
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.hooks.base import BaseHook
from datetime import datetime, timedelta
import json
import sys

dag_folder = os.path.dirname(os.path.abspath(__file__))
if dag_folder not in sys.path:
    sys.path.insert(0, dag_folder)

# Import the parsing methods
from parsing_methods.mistralparsing import process_pdf as mistral_process_pdf
from parsing_methods.doclingparsing import main as docling_process_pdf

# Load configuration from JSON file
with open('/opt/airflow/config/nvidia_config.json') as config_file:
    config = json.load(config_file)

# S3 Configuration
AWS_CONN_ID = config['AWS_CONN_ID']
BUCKET_NAME = config['BUCKET_NAME']
S3_SOURCE_FOLDER = config['S3_BASE_FOLDER']  
S3_DOCLING_OUTPUT = config['S3_DOCLING_OUTPUT'] 
S3_MISTRAL_OUTPUT = config['S3_MISTRAL_OUTPUT'] 

# DAG default arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=5)
}

# Create DAG
dag = DAG(
    "nvidia_pdf_to_markdown_pipeline",
    default_args=default_args,
    description="Convert NVIDIA quarterly PDF reports to markdown using Docling and Mistral",
    schedule_interval='@daily',
    catchup=False,
)

def list_pdf_files(**context):
    """List all PDF files in the S3 source folder structure"""
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    all_files = []
    
    # List all files in the source folder recursively
    keys = s3_hook.list_keys(bucket_name=BUCKET_NAME, prefix=S3_SOURCE_FOLDER)
    
    # Filter for PDF files
    pdf_files = [key for key in keys if key.endswith('.pdf')]
    
    # Organize the PDF files by year and quarter
    organized_files = {}
    
    for pdf_file in pdf_files:
        # Extract year and quarter from path pattern "nvidia_quarterly_reports/YYYY/QX.pdf"
        path_parts = pdf_file.split('/')
        if len(path_parts) >= 3:
            year = path_parts[-2]
            quarter = path_parts[-1].split('.')[0]
            
            if year not in organized_files:
                organized_files[year] = {}
                
            organized_files[year][quarter] = pdf_file
    
    print(f"Found PDF files: {organized_files}")
    return organized_files

def convert_pdfs_docling(**context):
    """Download PDFs from S3 and convert to markdown using Docling"""
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    organized_files = context['task_instance'].xcom_pull(task_ids='list_pdf_files')
    results = []
    
    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()
    try:
        for year, quarters in organized_files.items():
            for quarter, s3_key in quarters.items():
                # Download PDF file to temporary directory
                local_pdf_path = os.path.join(temp_dir, f"{quarter}.pdf")
                s3_hook.download_file(
                    key=s3_key,
                    bucket_name=BUCKET_NAME,
                    local_path=local_pdf_path
                )
                print(f"Downloaded {s3_key} to {local_pdf_path}")
                
                # Process the PDF using Docling
                try:
                    # The Docling function will save the markdown file in the output directory
                    docling_process_pdf(local_pdf_path)
                    
                    # Docling saves the file with a specific naming pattern in the output directory
                    output_dir = Path(f"output/{Path(local_pdf_path).stem}")
                    md_filename = f"{Path(local_pdf_path).stem}-with-images.md"
                    markdown_path = output_dir / md_filename
                    
                    if markdown_path.exists():
                        # Create the S3 destination key
                        s3_dest_key = f"{S3_DOCLING_OUTPUT}/{year}/{quarter}.md"
                        
                        # Upload the markdown file to S3
                        s3_hook.load_file(
                            filename=str(markdown_path),
                            key=s3_dest_key,
                            bucket_name=BUCKET_NAME,
                            replace=True
                        )
                        
                        print(f"Uploaded Docling markdown for {year}/{quarter} to {s3_dest_key}")
                        results.append({"year": year, "quarter": quarter, "method": "docling", "status": "success"})
                    else:
                        print(f"Error: Docling markdown file not found at {markdown_path}")
                        results.append({"year": year, "quarter": quarter, "method": "docling", "status": "error"})
                        
                except Exception as e:
                    print(f"Error processing {local_pdf_path} with Docling: {str(e)}")
                    results.append({"year": year, "quarter": quarter, "method": "docling", "status": "error", "error": str(e)})
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    return results

def convert_pdfs_mistral(**context):
    """Download PDFs from S3 and convert to markdown using Mistral"""
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    organized_files = context['task_instance'].xcom_pull(task_ids='list_pdf_files')
    results = []
    
    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()
    try:
        for year, quarters in organized_files.items():
            for quarter, s3_key in quarters.items():
                # Download PDF file to temporary directory
                local_pdf_path = os.path.join(temp_dir, f"{quarter}.pdf")
                s3_hook.download_file(
                    key=s3_key,
                    bucket_name=BUCKET_NAME,
                    local_path=local_pdf_path
                )
                print(f"Downloaded {s3_key} to {local_pdf_path}")
                
                # Create output directory for this file
                pdf_base = f"extracted_{year}_{quarter}"
                output_dir = os.path.join(temp_dir, "ocr_output", pdf_base)
                os.makedirs(output_dir, exist_ok=True)
                
                # Process the PDF using Mistral
                try:
                    mistral_process_pdf(local_pdf_path)
                    
                    # Mistral saves the output markdown in a specific location
                    markdown_path = os.path.join(output_dir, "output.md")
                    
                    if os.path.exists(markdown_path):
                        # Create the S3 destination key
                        s3_dest_key = f"{S3_MISTRAL_OUTPUT}/{year}/{quarter}.md"
                        
                        # Upload the markdown file to S3
                        s3_hook.load_file(
                            filename=markdown_path,
                            key=s3_dest_key,
                            bucket_name=BUCKET_NAME,
                            replace=True
                        )
                        
                        print(f"Uploaded Mistral markdown for {year}/{quarter} to {s3_dest_key}")
                        results.append({"year": year, "quarter": quarter, "method": "mistral", "status": "success"})
                    else:
                        print(f"Error: Mistral markdown file not found at {markdown_path}")
                        results.append({"year": year, "quarter": quarter, "method": "mistral", "status": "error"})
                
                except Exception as e:
                    print(f"Error processing {local_pdf_path} with Mistral: {str(e)}")
                    results.append({"year": year, "quarter": quarter, "method": "mistral", "status": "error", "error": str(e)})
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    return results

# Define the DAG tasks
list_files_task = PythonOperator(
    task_id='list_pdf_files',
    python_callable=list_pdf_files,
    provide_context=True,
    dag=dag,
)

docling_conversion_task = PythonOperator(
    task_id='docling_conversion',
    python_callable=convert_pdfs_docling,
    provide_context=True,
    dag=dag,
)

mistral_conversion_task = PythonOperator(
    task_id='mistral_conversion',
    python_callable=convert_pdfs_mistral,
    provide_context=True,
    dag=dag,
)

# Set up task dependencies
list_files_task >> [docling_conversion_task, mistral_conversion_task]

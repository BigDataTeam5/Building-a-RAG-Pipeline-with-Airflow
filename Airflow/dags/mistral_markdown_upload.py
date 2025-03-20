import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# DAG default arguments - keep minimal imports at top level
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}

# Create DAG
dag = DAG(
    "mistral_markdown_upload",
    default_args=default_args,
    description="Convert NVIDIA quarterly PDF reports to markdown using Docling and Mistral",
    schedule_interval='@daily',
    catchup=False,
)
def _load_config():
    """Load configuration from JSON file"""
    import json
    with open('/opt/airflow/config/nvidia_config.json') as config_file:
        return json.load(config_file)


# Functions remain the same...
# Only modifying the convert_pdfs_mistral function and task dependencies
def list_pdf_files(**context):
    """List all PDF files in the S3 source folder structure"""
    # Import S3Hook here instead of at top level
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook
    import re
    from dotenv import load_dotenv
    load_dotenv('/opt/airflow/.env')
    
    config = _load_config()
    AWS_CONN_ID = config['AWS_CONN_ID']
    BUCKET_NAME = config['BUCKET_NAME'].strip()
    
    # Validate bucket name format
    s3_bucket_pattern = re.compile(r'^[a-zA-Z0-9.\-_]{1,255}$')
    if not s3_bucket_pattern.match(BUCKET_NAME):
        print(f"WARNING: Invalid bucket name format: '{BUCKET_NAME}'. S3 bucket names must only contain letters, numbers, hyphens, underscores, and periods.")
        # Clean the bucket name by replacing invalid characters
        BUCKET_NAME = re.sub(r'[^a-zA-Z0-9.\-_]', '-', BUCKET_NAME)
        print(f"Auto-corrected bucket name to: '{BUCKET_NAME}'")
    
    S3_SOURCE_FOLDER = config['S3_BASE_FOLDER'].strip()
    print(f"Using S3 configuration: Bucket={BUCKET_NAME}, Source={S3_SOURCE_FOLDER}")
    
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    
    try:
        # List all files in the source folder recursively
        print(f"Attempting to list keys in bucket '{BUCKET_NAME}' with prefix '{S3_SOURCE_FOLDER}'")
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
        
        # Store config in XCom for downstream tasks
        context['ti'].xcom_push(key='config', value={
            'bucket_name': BUCKET_NAME,
            'aws_conn_id': AWS_CONN_ID,
            'docling_output': config['S3_DOCLING_OUTPUT'].strip(),
            'mistral_output': config['S3_MISTRAL_OUTPUT'].strip()
        })
        
        return organized_files
    except Exception as e:
        print(f"Error listing files from S3: {str(e)}")
        # Return empty dict to allow the DAG to continue
        return {}

def convert_pdfs_mistral(**context):
    """Download PDFs from S3 and convert to markdown using Mistral"""
    import tempfile
    import shutil
    import sys
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    # Add paths for importing custom modules
    dag_folder = os.path.dirname(os.path.abspath(__file__))
    if dag_folder not in sys.path:
        sys.path.insert(0, dag_folder)

    # Check if mistralai package is installed
    try:
        from parsing_methods.mistralparsing import process_pdf as mistral_process_pdf
        mistral_available = True
    except ImportError as e:
        print(f"Mistral AI package not installed: {str(e)}")
        print("Using fallback method...")
        mistral_available = False

    # Get config from XCom
    config = context['ti'].xcom_pull(key='config', task_ids='list_pdf_files')
    if not config:
        print("No configuration found. Cannot proceed.")
        return []

    BUCKET_NAME = config['bucket_name']
    AWS_CONN_ID = config['aws_conn_id']
    S3_MISTRAL_OUTPUT = config['mistral_output']

    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    organized_files = context['task_instance'].xcom_pull(task_ids='list_pdf_files')
    results = []

    if not organized_files:
        print("No PDF files found to process")
        return []

    if not mistral_available:
        print("Mistral OCR not available. Skipping processing.")
        for year, quarters in organized_files.items():
            for quarter, _ in quarters.items():
                results.append({
                    "year": year, 
                    "quarter": quarter, 
                    "method": "mistral", 
                    "status": "skipped", 
                    "error": "Mistral AI package not installed"
                })
        return results

    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()
    try:
        for year, quarters in organized_files.items():
            for quarter, s3_key in quarters.items():
                # Create a directory for this file
                file_dir = os.path.join(temp_dir, f"{year}_{quarter}")
                os.makedirs(file_dir, exist_ok=True)

                # Download PDF file to temporary directory
                local_pdf_path = os.path.join(file_dir, f"{quarter}.pdf")
                try:
                    print(f"Downloading {s3_key} to {local_pdf_path}")
                    s3_obj = s3_hook.get_key(key=s3_key, bucket_name=BUCKET_NAME)
                    with open(local_pdf_path, 'wb') as f:
                        f.write(s3_obj.get()['Body'].read())
                    print(f"Successfully downloaded {s3_key} to {local_pdf_path}")

                    # Process the PDF using Mistral with direct file upload
                    # Only pass the local file path - no S3 dependencies
                    markdown_path = mistral_process_pdf(local_pdf_path)
                    
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
                        results.append({
                            "year": year, 
                            "quarter": quarter, 
                            "method": "mistral", 
                            "status": "success"
                        })
                    else:
                        print(f"Error: Mistral markdown file not found at {markdown_path}")
                        results.append({
                            "year": year, 
                            "quarter": quarter, 
                            "method": "mistral", 
                            "status": "error"
                        })

                except Exception as e:
                    print(f"Error downloading {s3_key}: {str(e)}")
                    results.append({
                        "year": year, 
                        "quarter": quarter, 
                        "method": "mistral", 
                        "status": "error", 
                        "error": f"Download error: {str(e)}"
                    })
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


mistral_conversion_task = PythonOperator(
    task_id='mistral_conversion',
    python_callable=convert_pdfs_mistral,
    provide_context=True,
    dag=dag,
)



# Set up task dependencies - parallel processing
list_files_task >> mistral_conversion_task
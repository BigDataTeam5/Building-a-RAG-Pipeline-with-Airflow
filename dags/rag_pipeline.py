from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from scripts.fetch_nvidia_reports import fetch_nvidia_reports
from scripts.pdf_parsing_s3 import process_pdfs
from scripts.embedding_generation_s3 import generate_embeddings
from scripts.store_embeddings_s3 import store_embeddings

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 3, 1),
    "retries": 1,
}

dag = DAG(
    "rag_pipeline",
    default_args=default_args,
    description="Automated RAG pipeline with Airflow",
    schedule_interval="@daily",
)

fetch_task = PythonOperator(
    task_id="fetch_nvidia_reports",
    python_callable=fetch_nvidia_reports,
    dag=dag,
)

parse_task = PythonOperator(
    task_id="parse_pdfs",
    python_callable=process_pdfs,
    op_kwargs={"parser_choice": "Docling"},  # Change to "Mistral OCR" if needed
    dag=dag,
)

embed_task = PythonOperator(
    task_id="generate_embeddings",
    python_callable=generate_embeddings,
    dag=dag,
)

store_task = PythonOperator(
    task_id="store_embeddings",
    python_callable=store_embeddings,
    dag=dag,
)

fetch_task >> parse_task >> embed_task >> store_task

import os
import time
import zipfile
import random
import requests
import pandas as pd
import os
import re
import time
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowFailException
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.hooks.base import BaseHook
import json


from datetime import datetime, timedelta

# =========================
# SELENIUM IMPORTS
# =========================
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from airflow.providers.http.operators.http import SimpleHttpOperator
# Load configuration from JSON file
with open('/opt/airflow/config/nvidia_config.json') as config_file:
    config = json.load(config_file)

# S3 Configuration
# Fetch AWS credentials from Airflow connection
AWS_CONN_ID = config['AWS_CONN_ID']
aws_creds = BaseHook.get_connection(AWS_CONN_ID)
BUCKET_NAME = config['BUCKET_NAME']
AWS_ACCESS_KEY = aws_creds.login  # AWS Key
AWS_SECRET_KEY = aws_creds.password  # AWS Secret
S3_BASE_FOLDER = config['S3_BASE_FOLDER']
RESULTS_FOLDER = config['RESULTS_FOLDER']
TEMP_DATA_FOLDER = config['TEMP_DATA_FOLDER']
BASE_URL = config['BASE_URL']  # Keep original BASE_URL from config
USER_AGENTS = config['USER_AGENTS']
# =========================
# DAG DEFAULT ARGS
# =========================
default_args = {
    "owner": config['default_args']['owner'],
    "depends_on_past": config['default_args']['depends_on_past'],
    "start_date": datetime.fromisoformat(config['default_args']['start_date']),
    "retries": config['default_args']['retries'],
    "retry_delay": timedelta(minutes=int(config['default_args']['retry_delay'].split(':')[1]))
}
# =========================
# CONSTANTS / CONFIG
# =========================
# Update these constants to create the right folder structure
DOWNLOAD_FOLDER = os.path.join(TEMP_DATA_FOLDER, "downloads")
ROOT_FOLDER = os.path.join(TEMP_DATA_FOLDER, "nvidia_quarterly_report")  # Main folder
YEAR_FOLDER = os.path.join(ROOT_FOLDER, "2024")  # Year subfolder
# Remove the line that appends "/financials" to BASE_URL
dag = DAG(
    "nvidia_quarterly_reports_pipeline",
    default_args=default_args,
    description="Use Selenium to scrape NVIDIA data, download, extract, and upload to S3",
    schedule_interval='@daily',  # Change to @daily if needed
    catchup=False,
)


def get_nvidia_quarterly_links(year):
    """
    Extracts exclusively the 10-K/10-Q PDF links from the quarter accordion for the given year.
    Returns a dictionary where keys are "Q1", "Q2", "Q3", "Q4" and values are lists of PDF URLs.
    """
    print(f"Using URL: {BASE_URL}")
    print(f"Getting quarterly 10-K/10-Q links for year: {year}")
    
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    quarterly_reports = {}
    try:
        print(f"Accessing {BASE_URL}")
        driver.get(BASE_URL)
        time.sleep(5)
        # Select the target year from the dropdown
        try:
            year_dropdown = driver.find_element(By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear")
            year_select = Select(year_dropdown)
            year_select.select_by_value(str(year))
            print(f"Selected year {year} from dropdown")
            time.sleep(3)
        except Exception as e:
            print(f"Error selecting year {year}: {str(e)}")
        
        # Locate all accordion items (each quarter)
        accordion_items = driver.find_elements(By.CSS_SELECTOR, "div.evergreen-accordion.evergreen-financial-accordion-item")
        print(f"Found {len(accordion_items)} accordion items on the page")
        
        for item in accordion_items:
            try:
                # Expand the accordion if not already expanded
                try:
                    toggle_button = item.find_element(By.CSS_SELECTOR, "button.evergreen-financial-accordion-toggle")
                    if toggle_button.get_attribute("aria-expanded") == "false":
                        toggle_button.click()
                        time.sleep(1)
                except Exception as e:
                    print("Could not expand accordion item: ", e)
                
                # Get the quarter title (e.g., "Fourth Quarter 2025")
                title_elem = item.find_element(By.CSS_SELECTOR, "span.evergreen-accordion-title")
                quarter_text = title_elem.text.strip()
                print(f"Processing accordion titled: '{quarter_text}'")
                
                # Determine quarter
                quarter = None
                if "Fourth Quarter" in quarter_text:
                    quarter = "Q4"
                elif "Third Quarter" in quarter_text:
                    quarter = "Q3"
                elif "Second Quarter" in quarter_text:
                    quarter = "Q2"
                elif "First Quarter" in quarter_text:
                    quarter = "Q1"
                else:
                    print(f"Could not determine quarter from title: {quarter_text}")
                    continue
                
                # Find all PDF links in this accordion item
                pdf_links = item.find_elements(By.CSS_SELECTOR, "a.evergreen-financial-accordion-attachment-PDF")
                print(f"Found {len(pdf_links)} PDF links in accordion for {quarter_text}")
                
                for link in pdf_links:
                    href = link.get_attribute("href")
                    if not href or not href.endswith(".pdf"):
                        continue
                    
                    # Try to obtain the text from the child span or fallback to aria-label
                    link_text = ""
                    try:
                        span = link.find_element(By.CSS_SELECTOR, "span.evergreen-link-text.evergreen-financial-accordion-link-text")
                        link_text = span.text.strip()
                    except Exception:
                        link_text = link.get_attribute("aria-label") or ""
                    
                    # Use a case-insensitive check for "10-K" or "10-Q"
                    if "10-k" in link_text.lower() or "10-q" in link_text.lower():
                        quarterly_reports.setdefault(quarter, []).append(href)
                        print(f"✅ Added {quarter} document: {href} (detected via text: '{link_text}')")
            except Exception as e:
                print(f"Error processing an accordion item: {str(e)}")
        
        print(f"Final quarterly 10-K/10-Q links for {year}: {quarterly_reports}")
        return quarterly_reports
    except Exception as e:
        print(f"❌ Error in scraping NVIDIA reports for {year}: {str(e)}")
        return {}
    finally:
        driver.quit()



def download_report(url_list, download_folder, filename):
    """
    Download a PDF report from the first URL in url_list using HTTP requests,
    and save it to the download_folder with the given filename.
    
    Parameters:
      - url_list: a list containing one or more URLs (the function will use the first one)
      - download_folder: the folder where the PDF should be saved
      - filename: the desired output filename (e.g., "q1.pdf")
      
    Returns:
      - True if the download and file write succeed, False otherwise.
    """
    # Get absolute path for the download folder
    abs_download_path = os.path.abspath(download_folder)
    print(f"Download folder absolute path: {abs_download_path}")
    
    try:
        if not url_list:
            print(f"❌ No download URLs provided for {filename}")
            return False
        
        # Use the first URL from the list
        url = url_list[0]
        print(f"⬇️ Downloading report from: {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            # Build the output file path using the provided filename
            file_path = os.path.join(download_folder, filename)
            
            # Write the PDF content to disk
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded report to {file_path}")
            return True
        else:
            print(f"❌ Failed to download report. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading report: {str(e)}")
        return False



# =========================
# MAIN AIRFLOW TASK
# =========================
def main_task(**context):
    """
    Main task that:
      1) Loops through years 2020-2025.
      2) Uses get_nvidia_quarterly_links to get links to quarterly reports for each year.
      3) Downloads the first 10-K/10-Q report for each quarter and saves it with a simple filename
         (e.g. q1.pdf, q2.pdf) in the year folder.
      4) Organizes files by year folder.
      5) Stores original URLs in XCom for direct processing.
    """
    year_range = range(2020, 2026)  # 2020 to 2025 inclusive
    
    # Create base directory structure
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(ROOT_FOLDER, exist_ok=True)
    
    all_successful_quarters = {}
    all_year_folders = {}
    all_quarterly_reports = {}  # Store all quarterly report URLs
    
    try:
        for year in year_range:
            year_str = str(year)
            print(f"📅 Processing year {year_str}")
            
            # Create year-specific folder
            year_folder = os.path.join(ROOT_FOLDER, year_str)
            os.makedirs(year_folder, exist_ok=True)
            all_year_folders[year_str] = year_folder
            
            # Get the quarterly report links for this year
            quarterly_reports = get_nvidia_quarterly_links(year_str)
            all_quarterly_reports[year_str] = quarterly_reports
            
            if not quarterly_reports:
                print(f"⚠️ No quarterly reports found for {year_str}, skipping to next year")
                continue
            
            print(f"✅ Processing quarterly reports for {year_str}: {list(quarterly_reports.keys())}")
            
            # Process each quarter (only download the first found file per quarter)
            for quarter, url_list in quarterly_reports.items():
                print(f"Processing download for {year_str} {quarter}")
                
                # Store the URL list in XCom for later direct access
                context['task_instance'].xcom_push(
                    key=f'report_urls_{year_str}_{quarter}',
                    value=url_list
                )
                
                # Define the target filename (e.g., "q1.pdf", "q2.pdf", etc.)
                filename = quarter.lower() + ".pdf"
                
                # Download using the first URL from the list
                if download_report([url_list[0]], DOWNLOAD_FOLDER, filename):
                    # After download, move the file from the download folder to the year folder
                    file_path = os.path.join(DOWNLOAD_FOLDER, filename)
                    if not os.path.exists(file_path):
                        print(f"⚠️ File {filename} not found in download folder for {quarter}")
                        continue
                    dst_path = os.path.join(year_folder, filename)
                    os.rename(file_path, dst_path)
                    print(f"✅ Moved and renamed file to {dst_path} for {year_str} {quarter}")
                    all_successful_quarters.setdefault(year_str, {})[quarter] = filename
                else:
                    print(f"❌ Download failed for {year_str} {quarter}")
        
        # Push folder info to XCom if needed
        context['task_instance'].xcom_push(key='year_folders', value=all_year_folders)
        context['task_instance'].xcom_push(key='successful_quarters_by_year', value=all_successful_quarters)
        context['task_instance'].xcom_push(key='quarterly_reports', value=all_quarterly_reports)
    
    except Exception as e:
        print(f"❌ Error in main task: {str(e)}")
        raise AirflowFailException(f"Main task failed: {str(e)}")
    

def upload_and_cleanup(**context):
    """Uploads all files from multiple year folders to S3 and deletes them after upload."""
    try:
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        uploaded_files = []

        # Pull the year folders and successful quarters from XCom
        year_folders = context['task_instance'].xcom_pull(task_ids='nvdia_scrape_download_extract_upload', key='year_folders')
        
        if not year_folders:
            print("⚠️ No year folders found for upload.")
            return

        # Clean the bucket name - remove any whitespace
        clean_bucket_name = BUCKET_NAME.strip() if isinstance(BUCKET_NAME, str) else BUCKET_NAME
        print(f"Using bucket name: '{clean_bucket_name}' (length: {len(clean_bucket_name)})")
        
        # Process each year folder
        for year, folder_path in year_folders.items():
            if not os.path.exists(folder_path):
                print(f"⚠️ Year folder {folder_path} not found, skipping")
                continue
                
            print(f"🚀 Processing year folder: {year} at {folder_path}")
            
            # Maintain folder structure in S3
            s3_folder = f"{S3_BASE_FOLDER}/{year}"

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".pdf"):
                    local_file_path = os.path.join(folder_path, file_name)

                    # Upload to S3
                    s3_key = f"{s3_folder}/{file_name}"
                    print(f"Uploading {file_name} to S3 at {s3_key}...")

                    s3_hook.load_file(
                        filename=local_file_path,
                        key=s3_key,
                        bucket_name=clean_bucket_name,
                        replace=True
                    )

                    uploaded_files.append(s3_key)
                    print(f"✅ Uploaded: {s3_key}")

            # After successful upload, delete the files in this year folder
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
            
            # Remove the year folder
            os.rmdir(folder_path)
            print(f"🗑️ Cleaned up year folder: {folder_path}")
        
        # Remove the root folder if empty
        if os.path.exists(ROOT_FOLDER) and not os.listdir(ROOT_FOLDER):
            os.rmdir(ROOT_FOLDER)
            print(f"🗑️ Removed empty root folder: {ROOT_FOLDER}")

        print(f"🎉 Upload and cleanup complete. Uploaded {len(uploaded_files)} files across {len(year_folders)} years.")

    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        raise

def convert_pdfs_docling(**context):
    """Process PDFs with Docling using direct URLs, downloading as needed"""
    import tempfile
    import shutil
    import sys
    import os
    import requests
    from pathlib import Path
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook
    
    if '/opt/airflow' not in sys.path:
        sys.path.insert(0, '/opt/airflow')
  
    from Backend.parsing_methods.doclingparsing import main as docling_process_pdf
    docling_available = True
    print("✅ Successfully imported Docling parsing module")
    # Load configuration
    with open('/opt/airflow/config/nvidia_config.json') as config_file:
        config = json.load(config_file)
    
    BUCKET_NAME = config['BUCKET_NAME'].strip()
    AWS_CONN_ID = config['AWS_CONN_ID']
    S3_DOCLING_OUTPUT = config.get('S3_DOCLING_OUTPUT', 'docling_markdowns')
    
    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    results = []
    
    # Get the quarterly report links from XCom
    quarterly_reports_by_year = context['task_instance'].xcom_pull(
        task_ids='nvdia_scrape_download_extract_upload', 
        key='successful_quarters_by_year'
    )
    
    if not quarterly_reports_by_year:
        print("No quarterly reports found. Cannot proceed with Docling processing.")
        return []
    
    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()
    try:
        for year, quarters in quarterly_reports_by_year.items():
            for quarter, filename in quarters.items():
                try:
                    # Get the original URL for this report
                    original_urls = context['task_instance'].xcom_pull(
                        task_ids='nvdia_scrape_download_extract_upload', 
                        key=f'report_urls_{year}_{quarter}'
                    )
                    
                    if not original_urls:
                        print(f"No URL found for {year} {quarter}, checking quarter data...")
                        # Try to get the URL from the quarterly_reports data
                        quarter_data = context['task_instance'].xcom_pull(
                            task_ids='nvdia_scrape_download_extract_upload', 
                            key='quarterly_reports'
                        )
                        if quarter_data and year in quarter_data and quarter in quarter_data[year]:
                            original_urls = quarter_data[year][quarter]
                        
                    if not original_urls:
                        # If still no URL, create one from the S3 object
                        print(f"No direct URL found for {year} {quarter}, using S3 path")
                        s3_key = f"{config['S3_BASE_FOLDER']}/{year}/{quarter.lower()}.pdf"
                        
                        # Get URL from S3
                        presigned_url = s3_hook.get_conn().generate_presigned_url(
                            'get_object',
                            Params={'Bucket': BUCKET_NAME, 'Key': s3_key},
                            ExpiresIn=3600  # 1 hour expiration
                        )
                        original_url = presigned_url
                    else:
                        # Use the first URL from the list if it's a list
                        original_url = original_urls[0] if isinstance(original_urls, list) else original_urls
                    
                    print(f"Processing {year} {quarter} with URL: {original_url}")
                    
                    # Create directory for this file
                    file_dir = os.path.join(temp_dir, f"{year}_{quarter}")
                    os.makedirs(file_dir, exist_ok=True)
                    
                    # Download the PDF from the URL
                    local_pdf_path = os.path.join(file_dir, f"{quarter}.pdf")
                    print(f"Downloading from {original_url} to {local_pdf_path}")
                    
                    response = requests.get(original_url, timeout=30)
                    if response.status_code == 200:
                        with open(local_pdf_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Successfully downloaded to {local_pdf_path}")
                    else:
                        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
                    
                    # Process the PDF using Docling
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
                        results.append({
                            "year": year, 
                            "quarter": quarter, 
                            "method": "docling", 
                            "status": "success"
                        })
                    else:
                        print(f"Error: Docling markdown file not found at {markdown_path}")
                        results.append({
                            "year": year, 
                            "quarter": quarter, 
                            "method": "docling", 
                            "status": "error"
                        })
                
                except Exception as e:
                    print(f"Error processing {year} {quarter} with Docling: {str(e)}")
                    results.append({
                        "year": year, 
                        "quarter": quarter, 
                        "method": "docling", 
                        "status": "error", 
                        "error": str(e)
                    })
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    return results

def convert_pdfs_mistral(**context):
    """Process PDFs using Mistral OCR directly from source URLs"""
    import tempfile
    import sys
    import os
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook
    if '/opt/airflow' not in sys.path:
        sys.path.insert(0, '/opt/airflow')
    # Add paths for importing custom modules
    dag_folder = os.path.dirname(os.path.abspath(__file__))
    if dag_folder not in sys.path:
        sys.path.insert(0, dag_folder)
    
    # Get the quarterly report links from XCom
    quarterly_reports_by_year = context['task_instance'].xcom_pull(
        task_ids='nvdia_scrape_download_extract_upload', 
        key='successful_quarters_by_year'
    )
    
    # Pull configuration information
    with open('/opt/airflow/config/nvidia_config.json') as config_file:
        config = json.load(config_file)
    
    BUCKET_NAME = config['BUCKET_NAME'].strip()
    AWS_CONN_ID = config['AWS_CONN_ID']
    S3_MISTRAL_OUTPUT = config.get('S3_MISTRAL_OUTPUT', 'mistral_markdowns')
    
    # Initialize S3 hook
    s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
    results = []
    

    from Backend.parsing_methods.mistralparsing import process_pdf as mistral_process_pdf
    mistral_available = True
    print("✅ Successfully imported Mistral parsing module")
    
    
    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Get the links from the original get_nvidia_quarterly_links function
        for year, quarters in context['task_instance'].xcom_pull(task_ids='nvdia_scrape_download_extract_upload', key='successful_quarters_by_year').items():
            for quarter, filename in quarters.items():
                try:
                    # Get the original URL for this report
                    original_urls = context['task_instance'].xcom_pull(
                        task_ids='nvdia_scrape_download_extract_upload', 
                        key=f'report_urls_{year}_{quarter}'
                    )
                    
                    if not original_urls:
                        print(f"No URL found for {year} {quarter}, checking quarter data...")
                        original_urls = quarterly_reports_by_year.get(year, {}).get(quarter, [])
                        
                    if not original_urls:
                        # If still no URL, we need to use the S3 file
                        print(f"No direct URL found for {year} {quarter}, using S3 path")
                        s3_key = f"{config['S3_BASE_FOLDER']}/{year}/{quarter.lower()}.pdf"
                        
                        # Create a presigned URL for the S3 object
                        presigned_url = s3_hook.get_conn().generate_presigned_url(
                            'get_object',
                            Params={'Bucket': BUCKET_NAME, 'Key': s3_key},
                            ExpiresIn=3600  # 1 hour expiration
                        )
                        original_url = presigned_url
                    else:
                        # Use the first URL from the list
                        original_url = original_urls[0] if isinstance(original_urls, list) else original_urls
                    
                    print(f"Processing {year} {quarter} with original URL: {original_url}")
                    
                    # Create output directory for this file
                    output_dir = os.path.join(temp_dir, f"{year}_{quarter}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Process the PDF directly from URL using Mistral
                    markdown_path = mistral_process_pdf(
                        pdf_url=original_url,
                        output_dir=output_dir
                    )
                    
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
                    print(f"Error processing {year} {quarter} with Mistral: {str(e)}")
                    results.append({
                        "year": year, 
                        "quarter": quarter, 
                        "method": "mistral", 
                        "status": "error", 
                        "error": str(e)
                    })
    
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
    
    return results



# Single operator for entire process
main_operator = PythonOperator(
    task_id="nvdia_scrape_download_extract_upload",
    python_callable=main_task,
    dag=dag,
)
upload_task = PythonOperator(
    task_id='upload_tsv_files_and_cleanup',
    python_callable=upload_and_cleanup,
    provide_context=True,
    dag=dag
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


# Set task dependencies
main_operator >> upload_task >> [docling_conversion_task,mistral_conversion_task]
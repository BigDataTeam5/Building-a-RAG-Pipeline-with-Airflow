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
BASE_URL = config['BASE_URL']
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

dag = DAG(
    "nvidia_quarterly_reports_pipeline",
    default_args=default_args,
    description="Use Selenium to scrape NVIDIA data, download, extract, and upload to S3",
    schedule_interval='@daily',  # Change to @daily if needed
    catchup=False,
)

# =========================
# HELPER FUNCTIONS
# =========================

def wait_for_downloads(download_folder, timeout=60):
    """
    Wait until downloads are complete (no *.crdownload files) or until timeout (seconds).
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if any(f.endswith(".crdownload") for f in os.listdir(download_folder)):
            time.sleep(2)  # still downloading
        else:
            print("✅ All downloads completed.")
            return True
    print("❌ Timeout: downloads did not complete.")
    return False

def get_nvidia_quarterly_links(year='2024'):
    """
    Enhanced function that specifically targets official Form 10-Q/10-K documents.
    Returns a dictionary of quarterly report URLs for seamless integration with download_report.
    """
    print(f"Using URL: {BASE_URL}")
    
    # Ensure results directory exists
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    # Configure Chrome options for headless operation
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
    
    try:
        quarterly_reports = {}
        
        # Navigate to the financial reports page
        print(f"Accessing {BASE_URL}")
        driver.get(BASE_URL)
        time.sleep(5)  # Wait for page to load
        
        # Save page source and screenshots for debugging
        with open('/opt/airflow/logs/nvidia_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        driver.save_screenshot('/opt/airflow/logs/nvidia_financial_page.png')
        
        # Execute JavaScript to navigate to year 2024 (slide index 3)
        try:
            driver.execute_script("""
                // Try to activate the 2024 slider (4th slide, index 3)
                let slickSlider = document.querySelector('.module-financial-table_body-year-container');
                if (slickSlider && typeof slickSlider.slick === 'function') {
                    slickSlider.slick.slickGoTo(3);
                }
            """)
            print("Executed JavaScript to navigate to 2024 slide")
            time.sleep(2)  # Wait for slide transition
        except Exception as e:
            print(f"JavaScript navigation failed: {str(e)}, continuing with direct element selection")

        # Look for the Form 10-Q/Form 10-K section
        print("Looking for Form 10-Q/Form 10-K section in the financial tables...")
        
        # Find the table row that contains "Form 10-Q/Form 10-K"
        form_sections = driver.find_elements(By.XPATH, "//div[contains(@class, 'module-financial-table_body-row') and contains(., 'Form 10-Q/Form 10-K')]")
        
        if form_sections:
            print(f"Found {len(form_sections)} Form 10-Q/Form 10-K sections")
            
            # Within this section, find the 2024 slide (data-slick-index="3")
            for section in form_sections:
                try:
                    # Look for the 2024 slide within this section
                    year_slide = section.find_element(By.XPATH, ".//div[@data-slick-index='3' and contains(@class, 'slick-slide')]")
                    print("Found 2024 slide in Form 10-Q/Form 10-K section")
                    
                    # Find all links with the specific class 'module-financial-table_link Form 10-Q/Form 10-K'
                    links = year_slide.find_elements(By.XPATH, ".//a[contains(@class, 'module-financial-table_link')]")
                    
                    if links:
                        print(f"Found {len(links)} official Form 10-Q/10-K links in the 2024 slide")
                        
                        # Process each link to determine which quarter it belongs to
                        for link in links:
                            href = link.get_attribute("href")
                            text = link.text.strip()
                            
                            if not href or not href.endswith(".pdf"):
                                continue
                                
                            print(f"Processing link: {text} - {href}")
                            
                            # Determine quarter from the link text or URL
                            quarter = None
                            if text.startswith("Q1"):
                                quarter = "Q1"
                            elif text.startswith("Q2"):
                                quarter = "Q2"
                            elif text.startswith("Q3"):
                                quarter = "Q3"
                            elif text.startswith("Q4"):
                                quarter = "Q4"
                                
                            # If we can't determine from text, use URL path
                            if not quarter:
                                if "/q1/" in href.lower():
                                    quarter = "Q1"
                                elif "/q2/" in href.lower():
                                    quarter = "Q2"
                                elif "/q3/" in href.lower():
                                    quarter = "Q3"
                                elif "/q4/" in href.lower():
                                    quarter = "Q4"
                            
                            if quarter:
                                # Skip files that are clearly supplementary
                                if ("commentary" in href.lower() or 
                                    "presentation" in href.lower() or 
                                    "trend" in href.lower()):
                                    print(f"Skipping supplementary document: {href}")
                                    continue
                                    
                                # Add to our results - maintain the expected list format
                                if quarter not in quarterly_reports:
                                    quarterly_reports[quarter] = []
                                quarterly_reports[quarter].append(href)
                                print(f"✅ Found official {quarter} document: {href}")
                                
                except Exception as e:
                    print(f"Error processing Form 10-Q/Form 10-K section: {str(e)}")
        
       
        print(f"Final quarterly reports: {quarterly_reports}")
        return quarterly_reports
    
    except Exception as e:
        print(f"❌ Error in scraping NVIDIA reports: {str(e)}")
    
    finally:
        driver.quit()

def download_report(url_list, download_folder, quarter):
    """Download the quarterly report from the first URL in the list"""
    # Create a new driver instance for this download
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")
    
    # Get absolute path for download folder
    abs_download_path = os.path.abspath(download_folder)
    print(f"Download folder absolute path: {abs_download_path}")
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    try:
        if not url_list:
            print(f"❌ No download URLs found for {quarter}")
            return False
            
        url = url_list[0]  # Take the first URL (most relevant one)
        
        # Direct download using requests
        print(f"⬇️ Downloading {quarter} report from: {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            # Create a filename based on the URL
            filename = url.split('/')[-1]
            file_path = os.path.join(download_folder, filename)
            
            # Save the PDF
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded {quarter} report to {file_path}")
            return True
        else:
            print(f"❌ Failed to download {quarter} report. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading report for {quarter}: {str(e)}")
        return False
    finally:
        driver.quit()

# =========================
# MAIN AIRFLOW TASK
# =========================
def main_task(**context):
    """
    Single main task that:
    1) Uses get_nvidia_quarterly_links to get links to 2024 quarterly reports
    2) Downloads the reports directly as PDFs
    3) Prepares data for the next task
    """
    year = "2024"  # Specifically targeting 2024 as requested
    
    # Create directory structure
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(ROOT_FOLDER, exist_ok=True)
    os.makedirs(YEAR_FOLDER, exist_ok=True)

    try:
        # Get the quarterly report links
        quarterly_reports = get_nvidia_quarterly_links(year)
        
        if not quarterly_reports:
            raise AirflowFailException("❌ Failed to find any quarterly reports on the NVIDIA website")
        
        print(f"✅ Processing {len(quarterly_reports)} quarterly reports for {year}: {quarterly_reports.keys()}")
        
        # Download each report
        successful_quarters = []
        for quarter, urls in quarterly_reports.items():
            print(f"Processing downloads for {quarter}")
            if download_report(urls, DOWNLOAD_FOLDER, quarter):
                successful_quarters.append(quarter)
                
                # Rename and move downloaded files to the year folder
                downloaded_files = os.listdir(DOWNLOAD_FOLDER)
                if not downloaded_files:
                    print(f"⚠️ No files found in download folder for {quarter}")
                    continue
                
                # Get the most recently downloaded file
                latest_file = sorted(
                    [f for f in downloaded_files if not f.endswith(".crdownload")],
                    key=lambda x: os.path.getmtime(os.path.join(DOWNLOAD_FOLDER, x)),
                    reverse=True
                )[0]
                
                # Create a simple filename: q1.pdf, q2.pdf, etc.
                new_filename = f"{quarter.lower()}.pdf"
                
                # Move and rename file
                src_path = os.path.join(DOWNLOAD_FOLDER, latest_file)
                dst_path = os.path.join(YEAR_FOLDER, new_filename)
                
                os.rename(src_path, dst_path)
                print(f"✅ Moved and renamed: {latest_file} → {new_filename}")
        
        # Push the year folder and successful quarters to XCom
        context['task_instance'].xcom_push(key='year_folder', value=YEAR_FOLDER)
        context['task_instance'].xcom_push(key='successful_quarters', value=successful_quarters)
        
    except Exception as e:
        print(f"❌ Error in main task: {str(e)}")
        raise AirflowFailException(f"Main task failed: {str(e)}")            

def upload_and_cleanup(**context):
    """Uploads all files from the year folder to S3 and deletes them after upload."""
    try:
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        uploaded_files = []

        # Pull the year folder and successful quarters from XCom
        year_folder = context['task_instance'].xcom_pull(task_ids='nvdia_scrape_download_extract_upload', key='year_folder')
        
        if not year_folder or not os.path.exists(year_folder):
            print("⚠️ Year folder not found for upload.")
            return

        # Clean the bucket name - remove any whitespace
        clean_bucket_name = BUCKET_NAME.strip() if isinstance(BUCKET_NAME, str) else BUCKET_NAME
        print(f"🚀 Processing folder: {year_folder}")
        print(f"Using bucket name: '{clean_bucket_name}' (length: {len(clean_bucket_name)})")
        
        # Maintain folder structure in S3
        s3_base_path = f"{S3_BASE_FOLDER}/nvidia_quarterly_report/2024"

        for file_name in os.listdir(year_folder):
            if file_name.endswith(".pdf"):
                local_file_path = os.path.join(year_folder, file_name)

                # Upload to S3 with proper path
                s3_key = f"{s3_base_path}/{file_name}"
                print(f"Uploading {file_name} to S3 at {s3_key}...")

                s3_hook.load_file(
                    filename=local_file_path,
                    key=s3_key,
                    bucket_name=clean_bucket_name,
                    replace=True
                )

                uploaded_files.append(s3_key)
                print(f"✅ Uploaded: {s3_key}")

        # After successful upload, clean up the folders
        for file in os.listdir(year_folder):
            os.remove(os.path.join(year_folder, file))
        os.rmdir(year_folder)  # Remove year folder
        os.rmdir(ROOT_FOLDER)  # Remove root folder
        print(f"🗑️ Cleaned up folders: {year_folder} and {ROOT_FOLDER}")

        print("🎉 Upload and cleanup complete. Files uploaded:", uploaded_files)

    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        raise

def upload_and_cleanup(**context):
    """Uploads all files from the year folder to S3 and deletes them after upload."""
    try:
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        uploaded_files = []

        # Pull the year folder and successful quarters from XCom
        year_folder = context['task_instance'].xcom_pull(task_ids='nvdia_scrape_download_extract_upload', key='year_folder')
        
        if not year_folder or not os.path.exists(year_folder):
            print("⚠️ Year folder not found for upload.")
            return

        # Clean the bucket name - remove any whitespace
        clean_bucket_name = BUCKET_NAME.strip() if isinstance(BUCKET_NAME, str) else BUCKET_NAME
        print(f"🚀 Processing folder: {year_folder}")
        print(f"Using bucket name: '{clean_bucket_name}' (length: {len(clean_bucket_name)})")
        
        s3_folder = f"{S3_BASE_FOLDER}/2024"

        for file_name in os.listdir(year_folder):
            if file_name.endswith(".pdf"):
                local_file_path = os.path.join(year_folder, file_name)

                # Upload to S3
                s3_key = f"{s3_folder}/{file_name}"
                print(f"Uploading {file_name} to S3 at {s3_key}...")

                s3_hook.load_file(
                    filename=local_file_path,
                    key=s3_key,
                    bucket_name=clean_bucket_name,  # Use cleaned bucket name
                    replace=True
                )

                uploaded_files.append(s3_key)
                print(f"✅ Uploaded: {s3_key}")

        # After successful upload, delete the folder
        for file in os.listdir(year_folder):
            os.remove(os.path.join(year_folder, file))  # Delete files
        os.rmdir(year_folder)  # Remove folder
        print(f"🗑️ Deleted folder: {year_folder}")

        print("🎉 Upload and cleanup complete. Files uploaded:", uploaded_files)

    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        raise

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


# Set task dependencies
main_operator >> upload_task
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
from airflow.hooks.base import BaseHook
import json


from datetime import datetime, timedelta

# =========================
# SELENIUM IMPORTS
# =========================
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

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
DOWNLOAD_FOLDER = os.path.join(TEMP_DATA_FOLDER, "downloads")
EXTRACTED_FOLDER = os.path.join(TEMP_DATA_FOLDER, "extracted")

dag = DAG(
    "selenium_sec_pipeline",
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

def get_nvidia_quarterly_links(year='2022'):
    """
    Scrapes NVIDIA's 'Quarterly Results' page for a specific year,
    returning a dictionary: { 'First Quarter 2022': [pdf_link1, pdf_link2, ...], ... }.
    """
    # 1) Configure Selenium (headless Chrome)
    chrome_options = Options()
    chrome_options.add_argument("--headless")        # remove if you want to see the browser
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    try:
        # 2) Navigate to Quarterly Results page
        driver.get("https://investor.nvidia.com/financial-info/quarterly-results/default.aspx")

        # 3) (Optional) Close cookie consent banner if present
        time.sleep(3)  # short pause to allow banner to load
        try:
            accept_btn = driver.find_element(By.ID, "truste-consent-button")
            accept_btn.click()
            time.sleep(2)
        except:
            pass  # no banner found or different ID

        # 4) Wait for the year dropdown and select the desired year
        wait = WebDriverWait(driver, 15)
        year_dropdown = wait.until(EC.presence_of_element_located((By.ID, "year")))
        year_options = year_dropdown.find_elements(By.TAG_NAME, "option")
        for opt in year_options:
            if opt.text.strip() == year:
                opt.click()
                break

        # small pause to let page reload for the selected year
        time.sleep(2)

        # 5) Locate the quarterly blocks and gather PDF links
        quarter_blocks = driver.find_elements(By.CSS_SELECTOR, ".module_item")
        results = {}
        for block in quarter_blocks:
            # Each block typically has a heading, e.g. "Fourth Quarter 2022"
            try:
                heading_el = block.find_element(By.CSS_SELECTOR, ".module_subtitle")
                heading = heading_el.text.strip()
            except:
                continue

            # skip if it doesn't match the desired year
            if year not in heading:
                continue

            # gather PDF links in this block
            pdf_links = []
            links = block.find_elements(By.TAG_NAME, "a")
            for link in links:
                href = link.get_attribute("href")
                if href and href.lower().endswith(".pdf"):
                    pdf_links.append(href)

            # store them keyed by the quarter heading
            if pdf_links:
                results[heading] = pdf_links

        return results

    finally:
        driver.quit()

def download_report(driver, url, download_folder, quarter):
    """Download the quarterly report from the given URL"""
    try:
        driver.get(url)
        print(f"⬇️ Starting download for Q{quarter} from: {url}")
        time.sleep(5)  # Allow time for page to load
        
        # Look for download link - this may need adjustment based on actual page structure
        download_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf') or contains(@href, '.xls') or contains(@href, '.xlsx')]")
        if not download_links:
            raise AirflowFailException(f"❌ No download links found for Q{quarter}")
        
        # Click the first available download link
        download_links[0].click()
        time.sleep(5)  # Allow time for download to start
        
        # Wait for download to complete
        download_success = wait_for_downloads(download_folder, timeout=60)
        if not download_success:
            raise AirflowFailException(f"❌ Download for Q{quarter} did not complete in time.")
            
        return True
    except Exception as e:
        print(f"❌ Error downloading report for Q{quarter}: {str(e)}")
        return False

# =========================
# MAIN AIRFLOW TASK
# =========================
def main_task(**context):
    """
    Single main task that:
    1) Sets up Selenium
    2) Uses get_nvidia_quarterly_reports to get links to 2022 quarterly reports
    3) Downloads the reports
    4) Prepares data for the next task
    """
    year = "2022"  # Specifically targeting 2022 as requested
    
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Setup automatic downloads
    prefs = {
        "download.default_directory": DOWNLOAD_FOLDER,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    # Add random user-agent
    chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # Get the quarterly report links
        quarterly_reports = get_nvidia_quarterly_links(driver, year)
        
        if not quarterly_reports:
            raise AirflowFailException(f"❌ No quarterly reports found for year {year}")
        
        print(f"✅ Found {len(quarterly_reports)} quarterly reports for {year}: {quarterly_reports.keys()}")
        
        # Create folders for each quarter
        quarter_folders = {}
        for quarter in quarterly_reports.keys():
            quarter_folder = os.path.join(EXTRACTED_FOLDER, f"{year}{quarter.lower()}")
            os.makedirs(quarter_folder, exist_ok=True)
            quarter_folders[quarter] = quarter_folder
        
        # Download each report
        successful_quarters = []
        for quarter, url in quarterly_reports.items():
            if download_report(driver, url, DOWNLOAD_FOLDER, quarter):
                successful_quarters.append(quarter)
                
                # Move downloaded files to the corresponding quarter folder
                for file_name in os.listdir(DOWNLOAD_FOLDER):
                    if not file_name.endswith(".crdownload"):  # Skip incomplete downloads
                        src_path = os.path.join(DOWNLOAD_FOLDER, file_name)
                        dst_path = os.path.join(quarter_folders[quarter], file_name)
                        os.rename(src_path, dst_path)
        
        # Push the extracted folder paths and year_quarters to XCom
        extracted_folders = [quarter_folders[q] for q in successful_quarters]
        year_quarters = [f"{year}{q.lower()}" for q in successful_quarters]
        
        context['task_instance'].xcom_push(key='extracted_folders', value=extracted_folders)
        context['task_instance'].xcom_push(key='year_quarters', value=year_quarters)
        
    finally:
        driver.quit()

def upload_and_cleanup(**context):
    """Uploads all tab-delimited .txt files from temp_data/YYYYqQ folders to S3 and deletes them after upload."""
    try:
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        uploaded_files = []
        uploaded_folders = []

        # Pull the extracted folder paths and year_quarters from XCom
        extracted_folders = context['task_instance'].xcom_pull(task_ids='selenium_scrape_download_extract_upload', key='extracted_folders')
        year_quarters = context['task_instance'].xcom_pull(task_ids='selenium_scrape_download_extract_upload', key='year_quarters')

        if not extracted_folders:
            print("⚠️ No extracted folders found for upload.")
            return

        for folder, year_quarter in zip(extracted_folders, year_quarters):
            local_folder_path = folder
            s3_folder = f"{S3_BASE_FOLDER}/{year_quarter}"

            print(f"🚀 Processing folder: {folder}")

            for file_name in os.listdir(local_folder_path):
                if file_name.endswith(".txt"):
                    local_file_path = os.path.join(local_folder_path, file_name)

                    # Upload to S3
                    s3_key = f"{s3_folder}/{file_name}"
                    print(f"Uploading {file_name} to S3 at {s3_key}...")

                    s3_hook.load_file(
                        filename=local_file_path,
                        key=s3_key,
                        bucket_name=BUCKET_NAME,
                        replace=True
                    )

                    uploaded_files.append(s3_key)
                    print(f"✅ Uploaded: {s3_key}")

            # After successful upload, delete the folder
            for file in os.listdir(local_folder_path):
                os.remove(os.path.join(local_folder_path, file))  # Delete files
            os.rmdir(local_folder_path)  # Remove folder
            uploaded_folders.append(folder)
            print(f"🗑️ Deleted folder: {local_folder_path}")

        print("🎉 Upload and cleanup complete for folders:", uploaded_folders)

    except Exception as e:
        print(f"❌ Error during upload: {str(e)}")
        raise





# Single operator for entire process
main_operator = PythonOperator(
    task_id="selenium_scrape_download_extract_upload",
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
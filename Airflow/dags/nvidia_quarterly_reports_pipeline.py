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

def get_nvidia_quarterly_links(year):
    """
    Enhanced function that extracts quarterly reports from NVIDIA's quarterly results page.
    Returns a dictionary of quarterly report URLs for seamless integration with download_report.
    """
    print(f"Using URL: {BASE_URL}")
    print(f"Getting quarterly links for year: {year}")
    
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

        # Select the year from the dropdown first
        try:
            # Find and select the appropriate year from the dropdown
            year_dropdown = driver.find_element(By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear")
            print(f"Found year dropdown: {year_dropdown.get_attribute('outerHTML')[:100]}")
            
            # Create a Select object to interact with the dropdown
            year_select = Select(year_dropdown)
            
            # Select the target year
            year_select.select_by_value(str(year))
            print(f"Selected year {year} from dropdown")
            time.sleep(3)  # Wait for page to update
            
            # Save updated page after year selection
            driver.save_screenshot(f'/opt/airflow/logs/nvidia_after_year_selection_{year}.png')
            
            # Find all accordion sections for the selected year
            accordion_items = driver.find_elements(By.XPATH, "//div[contains(@class, 'evergreen-accordion-item')]")
            print(f"Found {len(accordion_items)} accordion items")
            
            # Process each quarter's accordion section
            for item in accordion_items:
                # Get the quarter title
                title_element = item.find_element(By.XPATH, ".//span[contains(@class, 'evergreen-accordion-title')]")
                quarter_title = title_element.text
                print(f"Processing accordion: {quarter_title}")
                
                # Determine which quarter (Q1, Q2, Q3, Q4)
                quarter = None
                if "First Quarter" in quarter_title:
                    quarter = "Q1"
                elif "Second Quarter" in quarter_title:
                    quarter = "Q2"
                elif "Third Quarter" in quarter_title:
                    quarter = "Q3"
                elif "Fourth Quarter" in quarter_title:
                    quarter = "Q4"
                
                if not quarter:
                    print(f"Could not determine quarter from title: {quarter_title}")
                    continue
                
                # Find 10-Q or 10-K links within this accordion section
                try:
                    # First ensure the accordion is expanded
                    toggle_button = item.find_element(By.XPATH, ".//button[contains(@class, 'evergreen-financial-accordion-toggle')]")
                    if "aria-expanded" not in toggle_button.get_attribute("outerHTML") or toggle_button.get_attribute("aria-expanded") == "false":
                        toggle_button.click()
                        time.sleep(1)  # Wait for expansion
                    
                    # Updated XPath selector to match the exact HTML structure
                    links = item.find_elements(By.XPATH, ".//a[contains(@class, 'evergreen-financial-accordion-link') and contains(@class, 'evergreen-link--text-with-icon') and (contains(., '10-Q') or contains(., '10-K'))]")
                    
                    print(f"Found {len(links)} potential 10-Q/10-K links in {quarter_title}")
                    
                    # If no links found with specific approach, try a more general selector
                    if not links:
                        links = item.find_elements(By.XPATH, ".//a[contains(@class, 'evergreen-financial-accordion-link')]")
                        print(f"Found {len(links)} links with general selector")
                    
                    for link in links:
                        href = link.get_attribute("href")
                        link_html = link.get_attribute("outerHTML")
                        text = link.text.strip()
                        
                        print(f"Examining link: Text=[{text}], href=[{href}]")
                        print(f"Link HTML (preview): {link_html[:200]}...")
                        
                        # Check link text for '10-Q' or '10-K' using span elements
                        link_text_elements = link.find_elements(By.XPATH, ".//span[@class='evergreen-link-text evergreen-financial-accordion-link-text']")
                        for text_element in link_text_elements:
                            element_text = text_element.text.strip()
                            print(f"Found link text element: {element_text}")
                            
                            if "10-Q" in element_text or "10-K" in element_text:
                                if href and href.endswith(".pdf"):
                                    if quarter not in quarterly_reports:
                                        quarterly_reports[quarter] = []
                                    quarterly_reports[quarter].append(href)
                                    print(f"✅ Added {quarter} document for {year}: {href}")
                                    break  # Take only the first 10-Q/10-K link per quarter
                
                except Exception as e:
                    print(f"Error finding 10-Q/10-K links in {quarter_title}: {str(e)}")
        
        except Exception as e:
            print(f"Error selecting year {year} from dropdown: {str(e)}")
            
        # If no reports found, try alternative approach - look directly for PDF links with correct patterns
        if not quarterly_reports:
            print(f"No quarterly reports found using dropdown approach, trying direct link search...")
            
            try:
                # Find all PDF links on the page
                pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
                
                for link in pdf_links:
                    href = link.get_attribute("href")
                    text = link.text.strip()
                    
                    # Check if this is a 10-Q/10-K report for the current year
                    if href and str(year) in href and ("10-Q" in text or "10-K" in text):
                        # Determine the quarter
                        quarter = None
                        if "Q1" in href or "First Quarter" in text:
                            quarter = "Q1"
                        elif "Q2" in href or "Second Quarter" in text:
                            quarter = "Q2"
                        elif "Q3" in href or "Third Quarter" in text:
                            quarter = "Q3"
                        elif "Q4" in href or "Fourth Quarter" in text:
                            quarter = "Q4"
                        
                        # If we can't determine from text, use URL
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
                            if quarter not in quarterly_reports:
                                quarterly_reports[quarter] = []
                            quarterly_reports[quarter].append(href)
                            print(f"✅ Added {quarter} document for {year} through direct search: {href}")
            
            except Exception as e:
                print(f"Error in direct link search approach: {str(e)}")
                
        print(f"Final quarterly reports for {year}: {quarterly_reports}")
        return quarterly_reports
    
    except Exception as e:
        print(f"❌ Error in scraping NVIDIA reports for {year}: {str(e)}")
        return {}
    
    finally:
        driver.quit()

def process_links(links, quarterly_reports, year):
    """Helper function to process links and add them to quarterly_reports dictionary"""
    for link in links:
        try:
            href = link.get_attribute("href")
            text = link.text.strip()
            
            if not href or not href.endswith(".pdf"):
                continue
                
            print(f"Processing link: {text} - {href}")
            
            # Determine quarter from the link text or URL
            quarter = None
            if text.startswith("Q1") or "Q1" in text:
                quarter = "Q1"
            elif text.startswith("Q2") or "Q2" in text:
                quarter = "Q2"
            elif text.startswith("Q3") or "Q3" in text:
                quarter = "Q3"
            elif text.startswith("Q4") or "Q4" in text:
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
                print(f"✅ Found official {quarter} document for {year}: {href}")
        except Exception as e:
            print(f"Error processing link: {e}")

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

def search_for_reports_by_pattern(year):
    """Search for quarterly reports using common patterns and timeframes"""
    reports = {}
    
    # NVIDIA follows a fiscal year pattern that's offset from calendar year
    # Their fiscal Q4 from prior year = calendar Q1 of current year
    
    # For 2024: Q4 FY2023 = Q1 2024, Q1 FY2024 = Q2 2024, etc.
    if str(year) == '2024':
        patterns = [
            # Q1 2024 (Jan-Mar) = Q4 FY2023 (ends Jan 2024)
            ("Q1", [f"{BASE_URL}/2023/q4/", ".pdf"]),
            # Q2 2024 (Apr-Jun) = Q1 FY2024
            ("Q2", [f"{BASE_URL}/2024/q1/", ".pdf"]),
            # Q3 2024 (Jul-Sep) = Q2 FY2024
            ("Q3", [f"{BASE_URL}/2024/q2/", ".pdf"]),
            # Q4 2024 (Oct-Dec) = Q3 FY2024 
            ("Q4", [f"{BASE_URL}/2024/q3/", ".pdf"])
        ]
    # For 2025: Q4 FY2024 = Q1 2025, Q1 FY2025 = Q2 2025, etc.
    elif str(year) == '2025':
        patterns = [
            # Q1 2025 (Jan-Mar) = Q4 FY2024 (ends Jan 2025)
            ("Q1", [f"{BASE_URL}/2024/q4/", ".pdf"]),
            # Q2 2025 (Apr-Jun) = Q1 FY2025
            ("Q2", [f"{BASE_URL}/2025/q1/", ".pdf"]),
            # Q3 2025 (Jul-Sep) = Q2 FY2025
            ("Q3", [f"{BASE_URL}/2025/q2/", ".pdf"]),
            # Q4 2025 (Oct-Dec) = Q3 FY2025
            ("Q4", [f"{BASE_URL}/2025/q3/", ".pdf"])
        ]
    else:
        return {}
    
    # Hard-coded commonly used document names and patterns from historical data
    common_filenames = [
        "10q.pdf",
        "10Q.pdf",
        "10k.pdf", 
        "10K.pdf",
        f"NVDA-{year}-Q1-10Q.pdf",
        f"NVDA-{year}-Q2-10Q.pdf", 
        f"NVDA-{year}-Q3-10Q.pdf",
        f"NVDA-{year}-Q4-10K.pdf",
        "form10q.pdf",
        "form10k.pdf"
    ]
    
    # Try the common patterns
    for quarter, (base_path, extension) in patterns:
        for filename in common_filenames:
            url = f"{base_path}{filename}"
            try:
                response = requests.head(url, timeout=3)
                if response.status_code == 200:
                    if quarter not in reports:
                        reports[quarter] = []
                    reports[quarter].append(url)
                    print(f"✅ Found {quarter} document for {year} using pattern search: {url}")
                    break  # Found one valid URL for this quarter
            except Exception:
                continue
    
    return reports

# =========================
# MAIN AIRFLOW TASK
# =========================
def main_task(**context):
    """
    Main task that:
    1) Loops through years 2020-2025
    2) Uses get_nvidia_quarterly_links to get links to quarterly reports for each year
    3) Downloads the reports directly as PDFs
    4) Organizes files by year folder
    """
    year_range = range(2020, 2026)  # 2020 to 2025 inclusive
    
    # Create base directory structure
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(ROOT_FOLDER, exist_ok=True)
    
    all_successful_quarters = {}
    all_year_folders = {}

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
            
            if not quarterly_reports:
                print(f"⚠️ No quarterly reports found for {year_str}, skipping to next year")
                continue
            
            print(f"✅ Processing {len(quarterly_reports)} quarterly reports for {year_str}: {quarterly_reports.keys()}")
            
            # Download each report for this year
            successful_quarters = []
            for quarter, urls in quarterly_reports.items():
                print(f"Processing downloads for {year_str} {quarter}")
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
                    dst_path = os.path.join(year_folder, new_filename)
                    
                    os.rename(src_path, dst_path)
                    print(f"✅ Moved and renamed: {latest_file} → {new_filename} for {year_str}")
            
            # Store successful quarters for this year
            if successful_quarters:
                all_successful_quarters[year_str] = successful_quarters
        
        # Push all year folders and successful quarters to XCom
        context['task_instance'].xcom_push(key='year_folders', value=all_year_folders)
        context['task_instance'].xcom_push(key='successful_quarters_by_year', value=all_successful_quarters)
        
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
import os
import time
import requests
import boto3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# AWS S3 Configuration
AWS_ACCESS_KEY = "AKIA2NK3YLXVCEEQO6SF"
AWS_SECRET_KEY = "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"
AWS_REGION = "us-east-2"
S3_BUCKET_NAME = "aibucket-riya"

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Create directory for PDFs
SAVE_DIR = "./data/raw_pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://investor.nvidia.com/financial-info/quarterly-results"

def upload_to_s3(file_path, bucket_name, s3_key):
    """Uploads a file to S3."""
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print(f"✅ Uploaded {s3_key} to S3 bucket {bucket_name}")
    except Exception as e:
        print(f"❌ Failed to upload {s3_key} to S3: {e}")

def fetch_nvidia_reports():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("detach", True)  # Keeps browser open after script ends

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 20)

    years = [2025, 2024, 2023, 2022, 2021]

    for year in years:
        print(f"\n🔹 Selecting Year: {year}")
        driver.get(BASE_URL)
        time.sleep(5)  # Allow page to load

        try:
            dropdowns = driver.find_elements(By.TAG_NAME, "select")
            dropdown = None
            for d in dropdowns:
                dropdown_id = d.get_attribute("id")
                if "Year" in dropdown_id or "selectEvergreenFinancialAccordionYear" in dropdown_id:
                    dropdown = d
                    break

            if not dropdown:
                print(f"⚠️ Year selection dropdown not found! Skipping year {year}...")
                continue

            wait.until(EC.element_to_be_clickable(dropdown))
            print(f"📌 Dropdown Found with ID: {dropdown.get_attribute('id')}")

            driver.execute_script("arguments[0].removeAttribute('hidden');", dropdown)
            driver.execute_script("arguments[0].style.display = 'block';", dropdown)
            time.sleep(2)

            select = Select(dropdown)
            available_years = [option.text for option in select.options]
            print(f"📌 Available Years in Dropdown: {available_years}")

            if str(year) not in available_years:
                print(f"⚠️ Year {year} not available in dropdown. Skipping...")
                continue

            driver.execute_script(f"arguments[0].value='{year}'; arguments[0].dispatchEvent(new Event('change'))", dropdown)
            time.sleep(5)  # Allow the page to update

            quarters = [
                ("Fourth Quarter", "10-K"),
                ("Third Quarter", "10-Q"),
                ("Second Quarter", "10-Q"),
                ("First Quarter", "10-Q")
            ]

            for quarter, doc_type in quarters:
                try:
                    print(f"🔍 Checking {quarter} {year} for {doc_type}...")
                    
                    quarter_xpath = f"//span[contains(@class, 'evergreen-accordion-title') and contains(text(), '{quarter} {year}')]"
                    header = wait.until(EC.presence_of_element_located((By.XPATH, quarter_xpath)))
                    
                    parent_button = header.find_element(By.XPATH, "./ancestor::button")
                    driver.execute_script("arguments[0].click();", parent_button)
                    time.sleep(5)  # Allow full expansion

                    pdf_xpath = f"//a[contains(@href, '.pdf') and contains(., '{doc_type}')]"
                    
                    wait.until(EC.presence_of_element_located((By.XPATH, pdf_xpath)))
                    pdf_elem = driver.find_element(By.XPATH, pdf_xpath)
                    pdf_url = pdf_elem.get_attribute("href")

                    if not pdf_url:
                        print(f"⚠️ No valid PDF link found for {quarter} {year}. Skipping...")
                        continue

                    filename = f"NVIDIA_{year}_{quarter.replace(' ','_')}_{doc_type}.pdf"
                    file_path = os.path.join(SAVE_DIR, filename)
                    print(f"📥 Downloading {filename} from {pdf_url}...")

                    response = requests.get(pdf_url)
                    if response.status_code == 200:
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                        print(f"✅ Saved: {filename}")

                        # Upload to S3
                        upload_to_s3(file_path, S3_BUCKET_NAME, f"nvidia_reports/{filename}")

                    else:
                        print(f"❌ Failed to download {filename}")

                except Exception as e:
                    print(f"⚠️ No PDF found for {quarter} {year}. Possible reason: {e}")

        except Exception as e:
            print(f"⚠️ Error selecting year {year}: {e}")

    driver.quit()

if __name__ == "__main__":
    fetch_nvidia_reports()

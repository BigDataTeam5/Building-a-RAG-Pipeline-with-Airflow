#!/usr/bin/env python3

import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from airflow.hooks.base import BaseHook
import json
# Load configuration from JSON file
with open('/opt/airflow/config/sec_config.json') as config_file:
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


def get_nvidia_quarterly_reports(year="2022"):
    """
    Navigate to the NVIDIA 'Quarterly Results' page, select the given 'year',
    and retrieve links for the 10-K (4th quarter) and 10-Q (other quarters).

    :param year: str, the year you want to scrape (e.g. "2022", "2021", etc.)
    :return: dict with quarter as key (Q1, Q2, Q3, Q4) and link to 10-K/10-Q as value
    """
    base_url = "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx"

    # -------------------------
    # 1) Setup Selenium driver
    # -------------------------
    chrome_options = Options()
    chrome_options.add_argument("--headless")       # run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    try:
        # -------------------------
        # 2) Open the target page
        # -------------------------
        driver.get(base_url)

        # Wait for the "Select Year" dropdown to be present
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "year"))
        )

        # -------------------------
        # 3) Select the desired year
        # -------------------------
        # The "Select Year" dropdown has an ID of "year". We'll pick the <option> that matches 'year'.
        year_dropdown = driver.find_element(By.ID, "year")
        year_options = year_dropdown.find_elements(By.TAG_NAME, "option")

        # Find the matching option and click it
        for opt in year_options:
            if opt.text.strip() == year:
                opt.click()
                break

        time.sleep(2)  # small pause to let the page refresh for the selected year

        # -------------------------
        # 4) Scrape the quarterly blocks
        # -------------------------
        # Each quarter block has headings like "Fourth Quarter 2022", "Third Quarter 2022", etc.
        # We'll gather them by their headings, then find the correct 10-K or 10-Q link within each block.
        reports_info = {}

        # Quarter blocks are in <div class="module_subtitle"> or <div class="module_data"> structure.
        # We'll fetch them by containing elements. Alternatively, we can look for headings by text.
        all_quarter_blocks = driver.find_elements(By.CSS_SELECTOR, ".module_item")

        for block in all_quarter_blocks:
            # The quarter heading is typically in <div class="module_subtitle"> with text "Fourth Quarter 2022", etc.
            try:
                quarter_heading = block.find_element(By.CSS_SELECTOR, ".module_subtitle").text
            except:
                continue  # skip if no heading

            # Example heading: "Fourth Quarter 2022"
            # We'll parse out which quarter this is.
            # You can also do something more robust with regex, but a simple split should work for now.
            # We only proceed if it matches the year we selected (just in case).
            if year not in quarter_heading:
                continue

            # Identify the quarter
            # e.g. "Fourth Quarter 2022" => "Fourth" => Q4
            if "Fourth" in quarter_heading:
                quarter = "Q4"
            elif "Third" in quarter_heading:
                quarter = "Q3"
            elif "Second" in quarter_heading:
                quarter = "Q2"
            elif "First" in quarter_heading:
                quarter = "Q1"
            else:
                continue  # not recognized

            # We want 10-K for Q4, 10-Q for Q1, Q2, Q3
            target_form = "10-K" if quarter == "Q4" else "10-Q"

            # -------------------------
            # 5) Within each block, find the link for the correct form
            # -------------------------
            links = block.find_elements(By.TAG_NAME, "a")
            # We'll look for a link with text that matches the target form.
            target_link = None
            for link in links:
                link_text = link.text.strip().upper()
                if target_form in link_text:
                    target_link = link.get_attribute("href")
                    break

            if target_link:
                reports_info[quarter] = target_link

        return reports_info

    finally:
        driver.quit()


def main():
    year = "2022"  # change as needed
    reports = get_nvidia_quarterly_reports(year=year)
    print(f"\n=== NVIDIA {year} Quarterly Reports ===")
    for qtr, link in reports.items():
        print(f"{qtr}: {link}")


if __name__ == "__main__":
    main()

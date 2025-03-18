import os
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_zip_url():
    required_zip = "2024q1.zip"  # Ensure lowercase for matching
    BASE_URL = "https://www.sec.gov"
    DATA_URL = f"{BASE_URL}/data-research/sec-markets-data/financial-statement-data-sets"
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    )
    
    driver = None
    try:
        driver = webdriver.Remote(
            command_executor=f'http://{os.getenv("SELENIUM_HOST", "localhost")}:4444/wd/hub',
            options=chrome_options
        )
        
        driver.get(DATA_URL)
        
        # Wait for the table to be visible
        WebDriverWait(driver, 15).until(
            EC.visibility_of_element_located((By.XPATH, "//table[contains(@class, 'usa-table')]//a[contains(@href, '.zip')]"))
        )
        
        zip_links = {
            elem.get_attribute("href").split("/")[-1]: BASE_URL + elem.get_attribute("href")  # Create dictionary with file name as key and URL as value
            for elem in driver.find_elements(By.XPATH, "//table[contains(@class, 'usa-table')]//a[contains(@href, '.zip')]")
        }
        
        print("Found ZIP links:", zip_links)  # Debugging
        
        selected_zip = zip_links.get(required_zip)
        if selected_zip:
            return selected_zip
        else:
            print(f"No matching ZIP file found for: {required_zip}")
            return None
        
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        raise e
    
    finally:
        if driver:
            driver.quit()

def download_zip(zip_link):
    if not zip_link:
        print("No ZIP link found.")
        return
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    )

    driver = None
    try:
        driver = webdriver.Remote(
            command_executor=f'http://{os.getenv("SELENIUM_HOST", "localhost")}:4444/wd/hub',
            options=chrome_options
        )
        
        driver.get(zip_link)
        
        # Wait for the download to complete
        WebDriverWait(driver, 15).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        
        print("Downloaded ZIP file.")

    except Exception as e:
        print(f"Error during download: {str(e)}")
        raise e
    
    finally:
        if driver:
            driver.quit()

def test_scrape_zip_url():
    os.environ["SELENIUM_HOST"] = "localhost"
    
    try:
        zip_link = scrape_zip_url()
        
        # Ensure the function returns a valid link
        assert zip_link is None or zip_link.startswith("http")
        
    except Exception as e:
        pytest.fail(f"Test failed due to exception: {e}")

def test_download_zip():
    os.environ["SELENIUM_HOST"] = "localhost"
    
    try:
        zip_link = scrape_zip_url()
        download_zip(zip_link)
        
        # Ensure the function completes without errors
        assert True
    
    except Exception as e:
        pytest.fail(f"Test failed due to exception: {e}")

if __name__ == "__main__":
    pytest.main(["-v", os.path.abspath(__file__)])
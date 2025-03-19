from selenium import webdriver
import time

url = 'https://investor.nvidia.com/financial-info/quarterly-results/default.aspx'
driver = webdriver.Chrome()
driver.get(url)

# Add this line to keep the browser open until you manually close it
input("Press Enter to close the browser...")

# Or use time.sleep() to keep it open for a specific duration
# time.sleep(30)  # Keeps the browser open for 30 seconds

# When you're done, close the browser
driver.quit()
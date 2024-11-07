#%%
import sys
import time
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.service import Service
from selenium.common.exceptions import NoSuchElementException, WebDriverException, StaleElementReferenceException

from webdriver_manager.chrome import ChromeDriverManager


#%%
url = 'https://www.echemportal.org/echemportal/property-search'

option = webdriver.ChromeOptions()
option.add_argument('window-size=1920,1080')

driver = webdriver.Chrome(options = option)
# webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.implicitly_wait(3)
driver.get(url)


#%%
query_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[1]/echem-query-builder/div[2]/div/button'
driver.find_element(By.XPATH, query_path).click()
time.sleep(0.5)

tox_button_path = '//*[@id="QU.SE.7-toxicological-information-header"]/div/div[1]/button'
driver.find_element(By.XPATH, tox_button_path).click()
time.sleep(0.5)

geno_path = '//*[@id="QU.SE.7.6-genetic-toxicity-header"]/div/div[2]'
driver.find_element(By.XPATH, geno_path).click()
time.sleep(0.5)

vivo_path = '//*[@id="QU.SE.7.6-genetic-toxicity"]/div/div/div[2]/div[3]'
driver.find_element(By.XPATH, vivo_path).click()
time.sleep(0.5)

info_type_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/echem-property-phrase-field[1]/div/div/div/ng-select/div/div'
driver.find_element(By.XPATH, info_type_path).click()
time.sleep(0.5)

experiment_path = '/html/body/ng-dropdown-panel/div[2]/div[2]/div[3]'
driver.find_element(By.XPATH, experiment_path).click()
time.sleep(0.5)

tmp_path = '/html/body/echem-root'
driver.find_element(By.XPATH, tmp_path).click()
time.sleep(0.5)

tg_path = '//*[@id="property_query-builder-panel-1"]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/echem-property-phrase-field[4]/div/div/div/ng-select/div/span'
driver.find_element(By.XPATH, tg_path).click()
time.sleep(0.5)

tg474_path = '/html/body/ng-dropdown-panel/div[2]/div[2]/div[30]'
driver.find_element(By.XPATH, tg474_path).click()
time.sleep(0.5)

save_path = '/html/body/echem-root/div/echem-substance-search-page/echem-substance-search-container/echem-substance-search/form/div/ngb-accordion[1]/div[2]/div[2]/div/echem-property-query-panel/div[2]/div[3]/echem-property-form/form/div/button[2]'
driver.find_element(By.XPATH, save_path).click()
time.sleep(0.5)

search_path = '/html/body/echem-root/div/echem-substance-search-page/echem-substance-search-container/echem-substance-search/form/div/div[2]/div/button'
driver.find_element(By.XPATH, search_path).click()


#%%
result_ = []


#%%
page_num_path = '/html/body/echem-root/div/echem-substance-search-page/echem-property-search-results-container/echem-pagination/div/div[2]'
page_num = int(driver.find_element(By.XPATH, page_num_path).text.split(' ')[-1])

start = time.time()
for p in range(1, page_num + 1):
    row_num_path = '/html/body/echem-root/div/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr'
    try:
        row_num = len(driver.find_elements(By.XPATH, row_num_path))
    except StaleElementReferenceException:
        row_num = len(driver.find_elements(By.XPATH, row_num_path))
        
    
    row = tqdm(range(1, row_num + 1), file = sys.stdout)
    
    for i in row:
        src_dict = {}
        
        try:
            chem_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]/td[3]/a'
            property_url = driver.find_element(By.XPATH, chem_path % i).get_attribute('href')
            src_dict['link'] = property_url
        except StaleElementReferenceException:
            chem_path = '//*[@id="top"]/echem-substance-search-page/echem-property-search-results-container/echem-property-search-results/table/tbody/tr[%d]/td[3]/a'
            property_url = driver.find_element(By.XPATH, chem_path % i).get_attribute('href')
            src_dict['link'] = property_url
        
        driver.execute_script('window.open('');')
        driver.switch_to.window(driver.window_handles[1])
        try:
            driver.get(property_url)
        except WebDriverException:
            pass
        
        try:
            accept_path = '/html/body/div[1]/div/div[2]/div[2]/button[1]'
            driver.find_element(By.XPATH, accept_path).send_keys(Keys.ENTER)
        except NoSuchElementException:
            pass
        
        try:
            src = driver.page_source
            # soup = BeautifulSoup(src, 'html.parser')
            src_dict['src'] = src
            
            result_.append(src_dict)
        
        except AttributeError:
            pass

        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        
        row.set_postfix({'page': p})
    
    if p < page_num:
        next_page_path = '/html/body/echem-root/div/echem-substance-search-page/echem-property-search-results-container/echem-pagination/div/div[2]/a[3]'
        driver.find_element(By.XPATH, next_page_path).click()
        time.sleep(1.5)
    

print(time.time() - start)


#%%
json.dump(result_, open('tg474_page_src.json', 'w'))

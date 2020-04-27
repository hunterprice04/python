#!/usr/bin/env
import requests
from bs4 import BeautifulSoup

# URL = 'https://inigo.williamson-tn.org/property_search/#'
# i = 201367
URL = 'https://archives.williamsoncounty-tn.gov/indexes/dataset/marriages'

# for i  in range(1,20000):


# URL = BASE_URL + str(i)
page = requests.get(URL)
print(page.content)
soup = BeautifulSoup(page.content,'html.parser')
results = soup.findall(id="groom_label")

# 
# for result in results:
# print(results)

import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from django.http import HttpResponse


def scrape_test(symbol):
    r = requests.get('https://www.moneycontrol.com/india/stockpricequote/')
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    mydivs = soup.findAll(["a", "title"], {"class": "bl_12"})
    links = []
    title = []

    for item in mydivs:
        links.append(item.get('href'))
        title.append(item.get('title'))

    stock_data = pd.DataFrame({'Link': links, 'Title': title})
    print(stock_data.shape)
    selected_company = ""
    for i, row in stock_data.iterrows():
        if row['Title'] == "3M India":
            selected_company = row['Link']
    print("Link of selected company: ", selected_company)
    time.sleep(2)

    r = requests.get(selected_company)
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    mydivs = soup.findAll("div", {"class": "value_txtfl"})
    myvalues = soup.findAll("div", {"class": "value_txtfr"})

    col_names = []
    col_names.append('Link')
    for div in mydivs:
        if div.text not in col_names:
            col_names.append(div.text)

    scraped_data = pd.DataFrame(columns=col_names)
    print(scraped_data.head())
    return HttpResponse(None, content_type='application/json')

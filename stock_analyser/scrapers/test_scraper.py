from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

from django.http import HttpResponse


def scrape_test(symbol):
    r = requests.get('https://www.moneycontrol.com/india/stockpricequote/')
    data = r.text
    soup = BeautifulSoup(data, 'html.parser')
    mydivs = soup.findAll(["a", "title"], {"class": "bl_12"})
    links = []
    title = []

    i = 0
    for item in mydivs:
        links.append(item.get('href'))
        print(item.get('title'))
        title.append(item.get('title'))

    # values = [link.get('href') for link in mydivs]
    stock_data = pd.DataFrame({'LINK': links, 'Title': title})
    print(stock_data.shape)
    print(stock_data.head(5))

    return HttpResponse(None, content_type='application/json')

import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from django.http import HttpResponse

from stock_analyser.helper.JsonHelper import dump_json


def get_cash_flows(data):
    cashflows = pd.DataFrame(columns=['Metric', 'Value'])
    soup = BeautifulSoup(data, 'html.parser')
    mydivs = soup.findAll("table", {"class": "mctable1 thborder sharePriceTotalCal"})
    for div in mydivs:
        print(div)
        if div.text == "Promoters":
            print(div)
    # line = mydivs.parent.find_next_sibling("td")
    # print(soup.find("wd_mobile", text="Operating Activities"))


    # i = 0
    # for div in test:
    #     cashflows._set_value(i, 'Metric', div.find("div", {"class": "value_txtfl"}).text)
    #     cashflows._set_value(i, 'Value', div.find("div", {"class": "value_txtfr"}).text)
    #     i = i + 1
    return cashflows


def scrape_test(symbol):
    selected_company = get_selected_company_link()
    print("Link of selected company: ", selected_company)
    time.sleep(2)

    r = requests.get(selected_company)
    data = r.text
    stand_alone_statistics = get_standalone_statistics(data)
    cashflows = get_cash_flows(data)
    final_response = {}
    for i, row in stand_alone_statistics.iterrows():
        if row["Metric"] in ['P/E', 'Industry P/E']:
            final_response[row["Metric"]] = row["Value"]

    response_json = dump_json(final_response)
    return HttpResponse(response_json, content_type='application/json')


def get_standalone_statistics(data):
    soup = BeautifulSoup(data, 'html.parser')
    mydivs = soup.find("div", {"id": "standalone_valuation"})
    test = mydivs.findAll("li", {"class": "clearfix"})
    stand_alone_statistics = pd.DataFrame(columns=['Metric', 'Value'])
    i = 0
    for div in test:
        stand_alone_statistics._set_value(i, 'Metric', div.find("div", {"class": "value_txtfl"}).text)
        stand_alone_statistics._set_value(i, 'Value', div.find("div", {"class": "value_txtfr"}).text)
        i = i + 1
    return stand_alone_statistics


def get_selected_company_link():
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
    return selected_company

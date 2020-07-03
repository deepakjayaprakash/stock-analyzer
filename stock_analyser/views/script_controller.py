import pandas as pd
import quandl as quandl
from django.http import JsonResponse

from stock_analyser import settings


def get_companies_stats(request):
    sql = "select industry, count(1) as count from companies group by industry order by count desc"
    df = pd.read_sql(sql, settings.DATABASE_URL)
    # print(df)
    industry_count = {}
    for i, row in df.iterrows():
        industry_count[row['industry']] = row['count']

    response = {}
    response['industry_count'] = industry_count

    return JsonResponse(response)


def get_company_details(request, name):
    response = {}
    name = "%%%%%s%%%%" % name
    sql = "select quandl_code, last_updated_at, actual_name, industry, symbol from companies where actual_name like '%s' or symbol " \
          "like '%s'" % (name, name)

    print(sql)
    df = pd.read_sql(sql, settings.DATABASE_URL)
    l = []
    for i, row in df.iterrows():
        item = {}
        item['quandl_code'] = row['quandl_code']
        item['last_updated_at'] = row['last_updated_at']
        item['actual_name'] = row['actual_name']
        item['industry'] = row['industry']
        item['symbol'] = row['symbol']
        l.append(item)

    response['companies'] = l
    return JsonResponse(response)


def get_company_data(request, name):
    df = quandl.get("BSE/BOM500209", authtoken=settings.QUANDL_AUTH_TOKEN, start_date="2020-06-01", end_date="2020-06-30")
    print(df.head())
    print(df.shape)
    print(df.columns)
    l = []
    for i, row in df.iterrows():
        item = {}
        item['Open'] = row['Open']
        item['Close'] = row['Close']
        item['No. of Shares'] = row['No. of Shares']
        item['Date'] = pd.to_datetime(i)
        l.append(item)
    response = {}
    response[name] = l
    return JsonResponse(response)
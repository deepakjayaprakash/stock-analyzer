import quandl as quandl
from django.http import JsonResponse

from stock_analyser import settings


def get_company_data(request, name):
    df = quandl.get("BSE/BOM500209", authtoken=settings.QUANDL_AUTH_TOKEN, start_date="2020-06-01",
                    end_date="2020-06-30")
    print(df.head())
    print(df.shape)
    print(df.columns)
    l = []
    for i, row in df.iterrows():
        item = {}
        item['Open'] = row['Open']
        item['Close'] = row['Close']
        item['No. of Shares'] = row['No. of Shares']
        item['Date'] = i
        l.append(item)
    response = {}
    response[name] = l
    return JsonResponse(response)


def load_company_data(request, id):
    df = quandl.get("BSE/BOM500209", authtoken=settings.QUANDL_AUTH_TOKEN, start_date="2020-06-01",
                    end_date="2020-06-30")
    print(df.head())
    print(df.shape)
    print(df.columns)
    l = []
    for i, row in df.iterrows():
        item = {}
        item['Open'] = row['Open']
        item['Close'] = row['Close']
        item['No. of Shares'] = row['No. of Shares']
        item['Date'] = i
        l.append(item)
    response = {}
    response[name] = l
    return JsonResponse(response)